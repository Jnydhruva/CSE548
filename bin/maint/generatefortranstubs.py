#!/usr/bin/env python
#!/bin/env python
#
#    Generates fortran stubs for PETSc using Sowings bfort program
#
import os
#
def FixFile(filename):
  ''' Fixes the C fortran stub files generated by bfort'''
  import re
  ff = open(filename)
  data = ff.read()
  ff.close()

  # gotta be a better way to do this
  data = re.subn('\nvoid ','\nPETSC_EXTERN void PETSC_STDCALL ',data)[0]
  data = re.subn('\nPetscErrorCode ','\nPETSC_EXTERN void PETSC_STDCALL ',data)[0]
  data = re.subn('Petsc([ToRm]*)Pointer\(int\)','Petsc\\1Pointer(void*)',data)[0]	
  data = re.subn('PetscToPointer\(a\) \(a\)','PetscToPointer(a) (*(PetscFortranAddr *)(a))',data)[0]
  data = re.subn('PetscFromPointer\(a\) \(int\)\(a\)','PetscFromPointer(a) (PetscFortranAddr)(a)',data)[0]
  data = re.subn('PetscToPointer\( \*\(int\*\)','PetscToPointer(',data)[0]
  data = re.subn('MPI_Comm comm','MPI_Comm *comm',data)[0]
  data = re.subn('\(MPI_Comm\)PetscToPointer\( \(comm\) \)','MPI_Comm_f2c(*(MPI_Fint*)(comm))',data)[0]
  data = re.subn('\(PetscInt\* \)PetscToPointer','',data)[0]
  data = re.subn('\(Tao\* \)PetscToPointer','',data)[0]
  data = re.subn('\(TaoConvergedReason\* \)PetscToPointer','',data)[0]
  data = re.subn('\(TaoLineSearch\* \)PetscToPointer','',data)[0]
  data = re.subn('\(TaoLineSearchConvergedReason\* \)PetscToPointer','',data)[0]
  match = re.compile(r"""\b(PETSC|TAO)(_DLL|VEC_DLL|MAT_DLL|DM_DLL|KSP_DLL|SNES_DLL|TS_DLL|FORTRAN_DLL)(EXPORT)""")
  data = match.sub(r'',data)

  ff = open(filename, 'w')
  ff.write('#include "petscsys.h"\n#include "petscfix.h"\n#include "petsc/private/fortranimpl.h"\n'+data)
  ff.close()

def FindSource(filename):
  import os.path
  gendir, fname = os.path.split(filename)
  base, ext = os.path.splitext(fname)
  sdir, ftn_auto = os.path.split(gendir)
  if ftn_auto != 'ftn-auto': return None # Something is wrong, skip
  sfname = os.path.join(sdir, base[:-1] + ext)
  return sfname
  sourcefile = FindSource(filename)
  if sourcefile and os.path.isfile(sourcefile):
    import shutil
    shutil.copystat(sourcefile, filename)
  return

def FixDir(petscdir,dir,verbose):
  ''' Fixes a directory of files generated by bfort.
      + Fixes the C stub files to be compilable C
      + Generates a makefile
      + copies over Fortran interface files that are generated'''
  mansec = 'unknown'
  cnames = []
  hnames = []
  parentdir =os.path.abspath(os.path.join(dir,'..'))
  for f in os.listdir(dir):
    ext = os.path.splitext(f)[1]
    if ext == '.c':
      FixFile(os.path.join(dir, f))
      cnames.append(f)
    elif ext == '.h90':
      hnames.append(f)
  if (cnames != [] or hnames != []):
    mfile=os.path.abspath(os.path.join(parentdir,'makefile'))
    try:
      fd=open(mfile,'r')
    except:
      print 'Error! missing file:', mfile
      return
    inbuf = fd.read()
    fd.close()
    cppflags = ""
    libbase = ""
    locdir = ""
    for line in inbuf.splitlines():
      if line.find('CPPFLAGS') >=0:
        cppflags = line
      if line.find('LIBBASE') >=0:
        libbase = line
      elif line.find('LOCDIR') >=0:
        locdir = line.rstrip() + 'ftn-auto/'
      elif line.find('SUBMANSEC') >=0:
        mansec = line.split('=')[1].lower().strip()
      elif line.find('MANSEC') >=0:
        mansec = line.split('=')[1].lower().strip()

    # now assemble the makefile
    outbuf  =  '\n'
    outbuf +=  "#requiresdefine   'PETSC_HAVE_FORTRAN'\n"
    outbuf +=  'ALL: lib\n'
    outbuf +=   cppflags + '\n'
    outbuf +=  'CFLAGS   =\n'
    outbuf +=  'FFLAGS   =\n'
    outbuf +=  'SOURCEC  = ' +' '.join(cnames)+ '\n'
    outbuf +=  'SOURCEF  =\n'
    outbuf +=  'SOURCEH  = ' +' '.join(hnames)+ '\n'
    outbuf +=  'DIRS     =\n'
    outbuf +=  libbase + '\n'
    outbuf +=  locdir + '\n'
    outbuf +=  'include ${PETSC_DIR}/lib/petsc/conf/variables\n'
    outbuf +=  'include ${PETSC_DIR}/lib/petsc/conf/rules\n'
    outbuf +=  'include ${PETSC_DIR}/lib/petsc/conf/test\n'

    ff = open(os.path.join(dir, 'makefile'), 'w')
    ff.write(outbuf)
    ff.close()

  # if dir is empty - remove it
  if os.path.exists(dir) and os.path.isdir(dir) and os.listdir(dir) == []:
    os.rmdir(dir)

  # save Fortran interface file generated (it is merged with others in a post-processing step)
  modfile = os.path.join(parentdir,'f90module.f90')
  if os.path.exists(modfile):
    if verbose: print 'Generating F90 interface for '+modfile
    fd = open(modfile)
    txt = fd.read()
    fd.close()
    if txt:
      if not os.path.isdir(os.path.join(petscdir,'include','petsc','finclude','ftn-auto')): os.mkdir(os.path.join(petscdir,'include','petsc','finclude','ftn-auto'))
      if not os.path.isdir(os.path.join(petscdir,'include','petsc','finclude','ftn-auto',mansec+'-tmpdir')): os.mkdir(os.path.join(petscdir,'include','petsc','finclude','ftn-auto',mansec+'-tmpdir'))
      fname =  os.path.join(petscdir,'include','petsc','finclude','ftn-auto',mansec+'-tmpdir',parentdir.replace('/','_')+'.h90')
      fd =open(fname,'w')
      fd.write(txt)
      fd.close()
    os.remove(modfile)


def PrepFtnDir(dir):
  ''' Generate a fnt-auto directory if needed'''
  import shutil
  if os.path.exists(dir) and not os.path.isdir(dir):
    raise RuntimeError('Error - specified path is not a dir: ' + dir)
  elif not os.path.exists(dir):
    os.mkdir(dir)
  else:
    files = os.listdir(dir)
    for file in files:
      if os.path.isdir(os.path.join(dir,file)): shutil.rmtree(os.path.join(dir,file))
      else: os.remove(os.path.join(dir,file))
  return

def processDir(arg,dirname,names):
  ''' Runs bfort on a directory and then fixes the files generated by bfort including moving generated F90 fortran interface files'''
  import commands
  petscdir = arg[0]
  bfort    = arg[1]
  verbose  = arg[2]
  newls = []
  outdir = os.path.join(dirname,'ftn-auto')

  for l in names:
    if os.path.splitext(l)[1] in ['.c','.h','.cu']:
      newls.append(l)
  if newls:
    PrepFtnDir(outdir)
    options = ['-dir '+outdir, '-mnative', '-ansi', '-nomsgs', '-noprofile', '-anyname', '-mapptr',
               '-mpi', '-shortargname', '-ferr', '-ptrprefix Petsc', '-ptr64 PETSC_USE_POINTER_CONVERSION',
               '-fcaps PETSC_HAVE_FORTRAN_CAPS', '-fuscore PETSC_HAVE_FORTRAN_UNDERSCORE',
               '-f90mod_skip_header','-f90modfile','f90module.f90']
    cmd = 'cd '+dirname+'; BFORT_CONFIG_PATH='+os.path.join(petscdir,'lib','petsc','conf')+' '+bfort+' '+' '.join(options+newls)
    (status,output) = commands.getstatusoutput(cmd)
    if status:
      raise RuntimeError('Error running bfort\n'+cmd+'\n'+output)
    try:
      FixDir(petscdir,outdir,verbose)
    except:
      print 'Error! with FixDir('+outdir+')'

  # remove from list of subdirectories all directories without source code
  rmnames=[]
  for name in names:
    if name in ['.git','.hg','SCCS', 'output', 'BitKeeper', 'examples', 'externalpackages', 'bilinear', 'ftn-auto','fortran','bin','maint','ftn-custom','config','f90-custom','ftn-kernels']:
      rmnames.append(name)
    # skip for ./configure generated $PETSC_ARCH directories
    if os.path.isdir(os.path.join(dirname,name,'lib','petsc')) or os.path.isdir(os.path.join(dirname,name,'lib','petsc-conf')) or os.path.isdir(os.path.join(dirname,name,'conf')):
      rmnames.append(name)
    # skip include/petsc directory
    if name == 'petsc':
      rmnames.append(name)
  for rmname in rmnames:
    if rmname in names: names.remove(rmname)
  return


def processf90interfaces(petscdir,verbose):
  ''' Takes all the individually generated fortran interface files and merges them into one for each mansec'''
  for mansec in os.listdir(os.path.join(petscdir,'include','petsc','finclude','ftn-auto')):
    if verbose: print 'Processing F90 interface for '+mansec
    if os.path.isdir(os.path.join(petscdir,'include','petsc','finclude','ftn-auto',mansec)):
      mansec = mansec[:-7]
      f90inc = os.path.join(petscdir,'include','petsc','finclude','ftn-auto','petsc'+mansec+'.h90')
      fd = open(f90inc,'w')
      for sfile in os.listdir(os.path.join(petscdir,'include','petsc','finclude','ftn-auto',mansec+'-tmpdir')):
        if verbose: print '  Copying in '+sfile
        fdr = open(os.path.join(petscdir,'include','petsc','finclude','ftn-auto',mansec+'-tmpdir',sfile))
        txt = fdr.read()
        fd.write(txt)
        fdr.close()
      fd.close()
      import shutil
      shutil.rmtree(os.path.join(petscdir,'include','petsc','finclude','ftn-auto',mansec+'-tmpdir'))
  FixDir(petscdir,os.path.join(petscdir,'include','petsc','finclude','ftn-auto'),verbose)
  return

def main(petscdir,bfort,dir,verbose):
  os.path.walk(dir, processDir, [petscdir, bfort,verbose])
  return
#
# generatefortranstubs bfortexectuable -verbose            -----  generates fortran stubs for a directory and all its children
# generatefortranstubs -merge  -verbose                    -----  merges fortran 90 interfaces definitions that have been generated
#
if __name__ ==  '__main__':
  import sys
  if len(sys.argv) < 2: sys.exit('Must give the BFORT program or -merge as the first argument')
  petscdir = os.environ['PETSC_DIR']
  if len(sys.argv) > 2: verbose = 1
  else: verbose = 0
  if not sys.argv[1].startswith('-'):
    main(petscdir,sys.argv[1],os.getcwd(),verbose)
  else:
    processf90interfaces(petscdir,verbose)

