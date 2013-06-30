
#include <petsc-private/tsimpl.h> /*I  "petscts.h" I*/

#undef __FUNCT__
#define __FUNCT__ "TSSetEventMonitor"
/*@C
   TSSetEventMonitor - Sets a monitoring function used for detecting events

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  nevents - number of local events
.  eventmonitor - event monitoring routine
.  postevent - [optional] post-event function
-  mectx - [optional] user-defined context for private data for the
              event monitor and post event routine (use NULL if no
              context is desired)

   Calling sequence of eventmonitor:
   PetscErrorCode EventMonitor(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,PetscInt *direction,PetscBool *terminate,void* mectx)

   Input Parameters:
+  ts  - the TS context
.  t   - current time
.  U   - current iterate
-  ctx - [optional] context passed with eventmonitor

   Output parameters:
+  fvalue    - function value of events at time t
.  direction - direction of zero crossing to be detected. -1 => Zero crossing in negative direction,
               +1 => Zero crossing in positive direction, 0 => both ways
-  terminate - terminate time stepping after event is detected.
               
   Calling sequence of postevent:
   PetscErrorCode PostEvent(TS ts,PetscInt nevents_zero, PetscInt events_zero, PetscReal t,Vec U,void* ctx)

   Input Parameters:
+  ts - the TS context
.  nevents_zero - number of local events whose event function is zero
.  events_zero  - indices of local events which have reached zero
.  t            - current time
.  U            - current solution
-  ctx          - the context passed with eventmonitor

   Level: intermediate

.keywords: TS, event, set, monitor

.seealso: TSCreate(), TSSetTimeStep(), TSSetConvergedReason()
@*/
PetscErrorCode TSSetEventMonitor(TS ts,PetscInt nevents,PetscErrorCode (*eventmonitor)(TS,PetscReal,Vec,PetscScalar*,PetscInt*,PetscBool*,void*),PetscErrorCode (*postevent)(TS,PetscInt,PetscInt[],PetscReal,Vec,void*),void *mectx)
{
  PetscErrorCode ierr;
  PetscReal      t;
  Vec            U;
  TSEvent        event;

  PetscFunctionBegin;
  ierr = PetscNew(struct _p_TSEvent,&ts->event);CHKERRQ(ierr);
  event = ts->event;
  ierr = PetscMalloc(nevents*sizeof(PetscScalar),&event->fvalue);CHKERRQ(ierr);
  ierr = PetscMalloc(nevents*sizeof(PetscScalar),&event->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscMalloc(nevents*sizeof(PetscBool),&event->terminate);CHKERRQ(ierr);
  ierr = PetscMalloc(nevents*sizeof(PetscInt),&event->direction);CHKERRQ(ierr);
  ierr = PetscMalloc(nevents*sizeof(PetscInt),&event->events_zero);CHKERRQ(ierr);
  event->monitor = eventmonitor;
  event->postevent = postevent;
  event->monitorcontext = (void*)mectx;
  event->nevents = nevents;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&event->initial_timestep);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  event->ptime_prev = t;
  ierr = (*event->monitor)(ts,t,U,event->fvalue_prev,event->direction,event->terminate,mectx);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TS Event options","");CHKERRQ(ierr);
  {
    event->tol = 1.0e-6;
    ierr = PetscOptionsReal("-ts_event_tol","","",event->tol,&event->tol,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSPostEvent"
PetscErrorCode TSPostEvent(TS ts,PetscInt nevents_zero,PetscInt events_zero[],PetscReal t,Vec U,void *ctx)
{
  PetscErrorCode ierr;
  TSEvent        event=ts->event;
  PetscBool      terminate=PETSC_FALSE;
  PetscInt       i;
  PetscBool      ts_terminate;

  PetscFunctionBegin;
  if (event->postevent) {
    ierr = (*event->postevent)(ts,nevents_zero,events_zero,t,U,ctx);CHKERRQ(ierr);
  }
  for(i = 0; i < nevents_zero;i++) {
    terminate = terminate || event->terminate[events_zero[i]];
  }
  ierr = MPI_Allreduce(&terminate,&ts_terminate,1,MPIU_INT,MPI_MAX,((PetscObject)ts)->comm);CHKERRQ(ierr);
  if (terminate) {
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_EVENT);CHKERRQ(ierr);
    event->status = TSEVENT_NONE;
  } else {
    event->status = TSEVENT_RESET_NEXTSTEP;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEventMonitorDestroy"
PetscErrorCode TSEventMonitorDestroy(TSEvent *event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*event)->fvalue);CHKERRQ(ierr);
  ierr = PetscFree((*event)->fvalue_prev);CHKERRQ(ierr);
  ierr = PetscFree((*event)->terminate);CHKERRQ(ierr);
  ierr = PetscFree((*event)->direction);CHKERRQ(ierr);
  ierr = PetscFree((*event)->events_zero);CHKERRQ(ierr);
  ierr = PetscFree(*event);CHKERRQ(ierr);
  *event = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEventMonitor"
PetscErrorCode TSEventMonitor(TS ts)
{
  PetscErrorCode ierr;
  TSEvent        event=ts->event;
  PetscReal      t;
  Vec            U;
  PetscInt       i;
  PetscReal      dt;
  PetscInt       status=event->status;
  PetscInt       rollback=0,in[2],out[2];

  PetscFunctionBegin;

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (event->status == TSEVENT_RESET_NEXTSTEP) {
    /* Take initial time step */
    dt = event->initial_timestep;
    ts->time_step = dt;
    event->status = TSEVENT_NONE;
  }

  if (event->status == TSEVENT_NONE) {
    event->tstepend   = t;
  }

  event->nevents_zero = 0;

  ierr = (*event->monitor)(ts,t,U,event->fvalue,event->direction,event->terminate,event->monitorcontext);CHKERRQ(ierr);
  if (event->status != TSEVENT_NONE) {
    for (i=0; i < event->nevents; i++) {
      if (PetscAbs(event->fvalue[i]) < event->tol) {
	event->status = TSEVENT_ZERO;
	event->events_zero[event->nevents_zero++] = i;
      }
    }
  }

  status = event->status;
  ierr = MPI_Allreduce(&status,&event->status,1,MPIU_INT,MPI_MAX,((PetscObject)ts)->comm);CHKERRQ(ierr);

  if (event->status == TSEVENT_ZERO) {
    dt = event->tstepend-t;
    ts->time_step = dt;
    ierr = TSPostEvent(ts,event->nevents_zero,event->events_zero,t,U,event->monitorcontext);CHKERRQ(ierr);
    for (i = 0; i < event->nevents; i++) {
      event->fvalue_prev[i] = event->fvalue[i];
    }
    event->ptime_prev  = t;
    PetscFunctionReturn(0);
  }

  for (i = 0; i < event->nevents; i++) {
    if ((event->direction[i] < 0 && PetscSign(event->fvalue[i]) <= 0 && PetscSign(event->fvalue_prev[i]) >= 0) || \
        (event->direction[i] > 0 && PetscSign(event->fvalue[i]) >= 0 && PetscSign(event->fvalue_prev[i]) <= 0) || \
        (event->direction[i] == 0 && PetscSign(event->fvalue[i])*PetscSign(event->fvalue_prev[i]) <= 0)) {

      event->status = TSEVENT_LOCATED_INTERVAL;
      rollback = 1;
      /* Compute linearly interpolated new time step */
      dt = PetscMin(dt,-event->fvalue_prev[i]*(t - event->ptime_prev)/(event->fvalue[i] - event->fvalue_prev[i]));
    }
  }
  in[0] = event->status;
  in[1] = rollback;
  ierr = MPI_Allreduce(in,out,2,MPIU_INT,MPI_MAX,((PetscObject)ts)->comm);CHKERRQ(ierr);
  
  rollback = out[1];
  if (rollback) {
    event->status = TSEVENT_LOCATED_INTERVAL;
  }

  if (event->status == TSEVENT_LOCATED_INTERVAL) {
    ierr = TSRollBack(ts);CHKERRQ(ierr);
    event->status = TSEVENT_PROCESSING;
  } else {
    for (i = 0; i < event->nevents; i++) {
      event->fvalue_prev[i] = event->fvalue[i];
    }
    event->ptime_prev  = t;
    if (event->status == TSEVENT_PROCESSING) {
      dt = event->tstepend - event->ptime_prev;
    }
  }
  PetscReal time_step;
  ierr = MPI_Allreduce(&dt,&time_step,1,MPIU_REAL,MPI_MIN,((PetscObject)ts)->comm);CHKERRQ(ierr);
  ts->time_step = time_step;
  PetscFunctionReturn(0);
}

