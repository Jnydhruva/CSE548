#ifndef PETSCMANAGEDTYPE_HPP
#define PETSCMANAGEDTYPE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpp/crtp.hpp>
#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/memory.hpp>
#include <petsc/private/cpp/functional.hpp>

#if defined(__cplusplus)
  #include <string>
  #include <iostream>
  #include <cstddef>
  #include <iterator>

namespace Petsc
{

struct copy_init_t { };
struct move_init_t { };
struct reference_init_t { };

namespace iterator
{

template <typename T>
class basic_pointer_iterator {
public:
  #if PETSC_CPP_VERSION >= 20
  using iterator_category = std::contiguous_iterator_tag;
  #else
  using iterator_category = std::random_access_iterator_tag;
  #endif
  using difference_type = std::ptrdiff_t;
  using value_type      = T;
  using pointer         = value_type *;
  using reference       = value_type &;

  // all basic_pointer_iterators are friends :)
  template <typename>
  friend class basic_pointer_iterator;

  constexpr explicit basic_pointer_iterator(pointer p = nullptr) noexcept : ptr_(p) { }
  constexpr basic_pointer_iterator(const basic_pointer_iterator &) noexcept            = default;
  constexpr basic_pointer_iterator(basic_pointer_iterator &&) noexcept                 = default;
  constexpr basic_pointer_iterator &operator=(const basic_pointer_iterator &) noexcept = default;
  constexpr basic_pointer_iterator &operator=(basic_pointer_iterator &&) noexcept      = default;

  // conversion from another pointer iterator, this also includes non-const-to-const
  // conversion!
  template <typename U, util::enable_if_t<std::is_convertible<U, value_type>::value, int> = 0>
  constexpr basic_pointer_iterator(const basic_pointer_iterator<U> &other) noexcept : ptr_(other.ptr_)
  {
  }

  PETSC_NODISCARD constexpr pointer base() const noexcept { return ptr_; }

  constexpr reference operator*() const noexcept { return *ptr_; }

  constexpr pointer operator->() noexcept { return ptr_; }

  // Prefix decrement
  constexpr basic_pointer_iterator &operator--() noexcept
  {
    --ptr_;
    return *this;
  }

  // Postfix decrement
  constexpr basic_pointer_iterator operator--(int) noexcept
  {
    basic_pointer_iterator tmp(*this);

    --(*this);
    return tmp;
  }

  // Prefix increment
  constexpr basic_pointer_iterator &operator++() noexcept
  {
    ++ptr_;
    return *this;
  }

  // Postfix increment
  constexpr basic_pointer_iterator operator++(int) noexcept
  {
    basic_pointer_iterator tmp(*this);

    ++(*this);
    return tmp;
  }

  constexpr basic_pointer_iterator &operator+=(difference_type diff) noexcept
  {
    ptr_ += diff;
    return *this;
  }

  constexpr basic_pointer_iterator &operator-=(difference_type diff) noexcept
  {
    ptr_ -= diff;
    return *this;
  }

  constexpr basic_pointer_iterator operator+(difference_type diff) const noexcept
  {
    basic_pointer_iterator tmp(*this);

    tmp += diff;
    return tmp;
  }

  constexpr basic_pointer_iterator operator-(difference_type diff) const noexcept { return *this + (-diff); }

private:
  pointer ptr_;
};

template <typename L, typename R>
constexpr typename basic_pointer_iterator<L>::difference_type operator-(basic_pointer_iterator<L> lhs, const basic_pointer_iterator<R> &rhs) noexcept
{
  return lhs.base() - rhs.base();
}

template <typename L, typename R>
constexpr bool operator==(const basic_pointer_iterator<L> &lhs, const basic_pointer_iterator<R> &rhs) noexcept
{
  return lhs.base() == rhs.base();
}

template <typename L, typename R>
constexpr bool operator!=(const basic_pointer_iterator<L> &lhs, const basic_pointer_iterator<R> &rhs) noexcept
{
  return !(lhs == rhs);
}

template <typename L, typename R>
constexpr bool operator<(const basic_pointer_iterator<L> &lhs, const basic_pointer_iterator<R> &rhs) noexcept
{
  return lhs.base() < rhs.base();
}

template <typename L, typename R>
constexpr bool operator>(const basic_pointer_iterator<L> &lhs, const basic_pointer_iterator<R> &rhs) noexcept
{
  return rhs < lhs;
}

template <typename L, typename R>
constexpr bool operator<=(const basic_pointer_iterator<L> &lhs, const basic_pointer_iterator<R> &rhs) noexcept
{
  return !(lhs > rhs);
}

template <typename L, typename R>
constexpr bool operator>=(const basic_pointer_iterator<L> &lhs, const basic_pointer_iterator<R> &rhs) noexcept
{
  return !(lhs < rhs);
}

template <typename T>
constexpr basic_pointer_iterator<T> operator+(typename basic_pointer_iterator<T>::difference_type diff, basic_pointer_iterator<T> rhs) noexcept
{
  rhs += diff;
  return rhs;
}

template <typename T>
constexpr basic_pointer_iterator<T> operator-(typename basic_pointer_iterator<T>::difference_type diff, basic_pointer_iterator<T> rhs) noexcept
{
  rhs -= diff;
  return rhs;
}

} // namespace iterator

namespace memory
{

// ==========================================================================================
// managed_storage
// ==========================================================================================

template <typename T>
class managed_storage {
public:
  using value_type     = T;
  using size_type      = std::size_t;
  using const_iterator = iterator::basic_pointer_iterator<const value_type>;
  using iterator       = iterator::basic_pointer_iterator<value_type>;

  managed_storage() noexcept = default;
  ~managed_storage() noexcept;

  managed_storage(PetscDeviceContext, PetscMemType, size_type) noexcept;

  template <typename Iterator>
  managed_storage(copy_init_t, PetscDeviceContext, PetscMemType, Iterator, Iterator) noexcept;
  template <typename Iterator>
  managed_storage(reference_init_t, PetscDeviceContext, PetscMemType, Iterator, Iterator) noexcept;
  template <typename Iterator>
  managed_storage(move_init_t, PetscDeviceContext, PetscMemType, Iterator, Iterator) noexcept;

  managed_storage(const managed_storage &) noexcept;
  managed_storage &operator=(const managed_storage &) noexcept;
  managed_storage(managed_storage &&) noexcept;
  managed_storage &operator=(managed_storage &&) noexcept;

  PETSC_NODISCARD value_type       *data() noexcept { return ptr_; }
  PETSC_NODISCARD const value_type *cdata() const noexcept { return ptr_; }
  PETSC_NODISCARD const value_type *data() const noexcept { return cdata(); }
  PETSC_NODISCARD bool              empty() const noexcept { return capacity() == 0; }
  PETSC_NODISCARD size_type         capacity() const noexcept { return capacity_; }
  PETSC_NODISCARD PetscMemType      mem_type() const noexcept { return mtype_; }

  PETSC_NODISCARD iterator begin() noexcept { return iterator{data()}; }
  PETSC_NODISCARD iterator end(size_type size) noexcept { return iterator{std::next(data(), size)}; }

  PETSC_NODISCARD const_iterator cbegin() const noexcept { return const_iterator{data()}; }
  PETSC_NODISCARD const_iterator cend(size_type size) const noexcept { return const_iterator{std::next(data(), size)}; }

  PETSC_NODISCARD const_iterator begin() const noexcept { return cbegin(); }
  PETSC_NODISCARD const_iterator end(size_type size) const noexcept { return cend(size); }

  PETSC_NODISCARD PetscErrorCode touch(PetscDeviceContext, PetscMemoryAccessMode) const noexcept;
  PETSC_NODISCARD PetscErrorCode reserve(PetscDeviceContext, size_type) noexcept;
  PETSC_NODISCARD PetscErrorCode destroy(PetscDeviceContext = nullptr) noexcept;
  template <typename Iterator>
  PETSC_NODISCARD PetscErrorCode assign(PetscDeviceContext, Iterator, Iterator, bool = false) noexcept;
  PETSC_NODISCARD PetscErrorCode assign(PetscDeviceContext, const managed_storage &) noexcept;

private:
  value_type  *ptr_      = nullptr;
  size_type    capacity_ = 0;
  PetscMemType mtype_    = PETSC_MEMTYPE_HOST;
  bool         own_ptr_  = true;

  // if this is true we can get a "free" own_ptr_ parameter without increasing the memory
  // footprint of the class since there is enough leftover padding due to size_type alignment
  static_assert((sizeof(PetscMemType) + sizeof(bool)) < sizeof(size_type), "");

  template <typename Iterator>
  constexpr managed_storage(Iterator begin, Iterator end, PetscMemType mtype, bool own = true) noexcept : ptr_(begin), capacity_(std::distance(begin, end)), mtype_(mtype), own_ptr_(own)
  {
  }

  template <typename Iterator>
  PETSC_NODISCARD PetscErrorCode assign_(PetscDeviceContext, Iterator, Iterator, size_type, std::random_access_iterator_tag) noexcept;
  // TODO
  template <typename Iterator>
  PETSC_NODISCARD PetscErrorCode assign_(PetscDeviceContext, Iterator, Iterator, size_type, ...) noexcept = delete;
};

// ==========================================================================================
// managed_storage - Private API
// ==========================================================================================

template <typename T>
template <typename Iterator>
inline PetscErrorCode managed_storage<T>::assign_(PetscDeviceContext dctx, Iterator begin, Iterator, size_type n, std::random_access_iterator_tag) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceArrayCopy(dctx, data(), std::addressof(*begin), n));
  PetscFunctionReturn(0);
}

// ==========================================================================================
// managed_storage - Public API
// ==========================================================================================

// size constructor
template <typename T>
inline managed_storage<T>::managed_storage(PetscDeviceContext, PetscMemType mtype, size_type n) noexcept : capacity_(n), mtype_(mtype)
{
}

// copy constructor
template <typename T>
template <typename Iterator>
inline managed_storage<T>::managed_storage(copy_init_t, PetscDeviceContext dctx, PetscMemType mtype, Iterator begin, Iterator end) noexcept : mtype_(mtype)
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, assign(dctx, std::move(begin), std::move(end)));
  PetscFunctionReturnVoid();
}

// use constructor
template <typename T>
template <typename Iterator>
inline managed_storage<T>::managed_storage(reference_init_t, PetscDeviceContext, PetscMemType mtype, Iterator begin, Iterator end) noexcept : managed_storage(std::move(begin), std::move(end), mtype, false)
{
}

// move constructor
template <typename T>
template <typename Iterator>
inline managed_storage<T>::managed_storage(move_init_t, PetscDeviceContext, PetscMemType mtype, Iterator begin, Iterator end) noexcept : managed_storage(std::move(begin), std::move(end), mtype)
{
}

template <typename T>
inline managed_storage<T>::~managed_storage() noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, this->destroy());
  PetscFunctionReturnVoid();
}

template <typename T>
inline managed_storage<T>::managed_storage(const managed_storage &other) noexcept : managed_storage(copy_init_t{}, nullptr, other.mem_type(), other.cbegin(), other.cend(other.capacity()))
{
}

template <typename T>
inline managed_storage<T> &managed_storage<T>::operator=(const managed_storage &other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) PetscCallAbort(PETSC_COMM_SELF, assign(nullptr, other));
  PetscFunctionReturn(*this);
}

template <typename T>
inline managed_storage<T>::managed_storage(managed_storage &&other) noexcept : ptr_(util::exchange(other.ptr_, nullptr)), capacity_(util::exchange(other.capacity_, 0)), mtype_(other.mtype_), own_ptr_(util::exchange(other.own_ptr_, true))
{
}

template <typename T>
inline managed_storage<T> &managed_storage<T>::operator=(managed_storage &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    // delete our pointer (if we have one)
    PetscAssertAbort(other.mem_type() == mem_type(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Memtypes don't match mine %s != theirs %s", PetscMemTypeToString(mem_type()), PetscMemTypeToString(other.mem_type()));
    PetscCallAbort(PETSC_COMM_SELF, destroy());
    ptr_      = util::exchange(other.ptr_, nullptr);
    capacity_ = util::exchange(other.capacity_, 0);
    own_ptr_  = util::exchange(other.own_ptr_, true);
  }
  PetscFunctionReturn(*this);
}

template <typename T>
inline PetscErrorCode managed_storage<T>::touch(PetscDeviceContext dctx, PetscMemoryAccessMode mode) const noexcept
{
  PetscPointerAttributes attr;

  PetscFunctionBegin;
  PetscCall(PetscDeviceGetPointerAttributes(data(), &attr));
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, attr.id, mode, nullptr));
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode managed_storage<T>::reserve(PetscDeviceContext dctx, size_type n) noexcept
{
  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  PetscCheck(own_ptr_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot reserve for a pointer that is not owned");
  if (data()) {
    if (capacity() < n) PetscCall(PetscDeviceRealloc(dctx, n, &ptr_));
  } else {
    PetscCall(PetscDeviceMalloc(dctx, mem_type(), n, &ptr_));
  }
  capacity_ = std::max(capacity(), n);
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode managed_storage<T>::destroy(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (ptr_ && own_ptr_) PetscCall(PetscDeviceFree(dctx, ptr_));
  ptr_      = nullptr;
  capacity_ = 0;
  own_ptr_  = true;
  PetscFunctionReturn(0);
}

template <typename T>
template <typename Iterator>
inline PetscErrorCode managed_storage<T>::assign(PetscDeviceContext dctx, Iterator begin, Iterator end, bool fast_assign) noexcept
{
  const auto n = std::distance(begin, end);

  PetscFunctionBegin;
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Begin < end");
  PetscCall(reserve(dctx, n));
  if (!n) PetscFunctionReturn(0);
  if (fast_assign) {
    std::copy(std::move(begin), std::move(end), this->begin());
    PetscCall(touch(dctx, PETSC_MEMORY_ACCESS_WRITE));
  } else {
    PetscCall(assign_(dctx, std::move(begin), std::move(end), n, typename std::iterator_traits<Iterator>::iterator_category{}));
  }
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode managed_storage<T>::assign(PetscDeviceContext dctx, const managed_storage &other) noexcept
{
  PetscFunctionBegin;
  PetscAssert(other.mem_type() == mem_type(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Memtypes don't match mine %s != theirs %s", PetscMemTypeToString(mem_type()), PetscMemTypeToString(other.mem_type()));
  PetscCall(assign(dctx, other.cbegin(), other.cend(other.capacity())));
  PetscFunctionReturn(0);
}

} // namespace memory

namespace expr
{

// ==========================================================================================
// ExpressionBase
// ==========================================================================================
template <typename D>
struct ExpressionBase : util::crtp<D, ExpressionBase> {
  using size_type = std::size_t;

  PETSC_NODISCARD size_type size() const noexcept;

  PETSC_NODISCARD auto expr_at(size_type) const noexcept;

  template <typename... Args>
  PETSC_NODISCARD PetscErrorCode prefetch(PetscDeviceContext, Args &&...) const noexcept;
};

template <typename D>
inline typename ExpressionBase<D>::size_type ExpressionBase<D>::size() const noexcept
{
  return this->underlying().size_impl_();
}

template <typename D>
inline auto ExpressionBase<D>::expr_at(size_type idx) const noexcept
{
  return this->underlying().at_impl_(idx);
}

template <typename D>
template <typename... Args>
inline PetscErrorCode ExpressionBase<D>::prefetch(PetscDeviceContext dctx, Args &&...args) const noexcept
{
  PetscFunctionBegin;
  PetscCall(this->underlying().prefetch_impl_(dctx, std::forward<Args>(args)...));
  PetscFunctionReturn(0);
}

// ==========================================================================================
// BinaryManagedExpression
// ==========================================================================================

template <typename L, typename R, typename F>
class BinaryManagedExpression : public ExpressionBase<BinaryManagedExpression<L, R, F>> {
  const L &lhs_;
  const R &rhs_;
  F        op_;

public:
  using base_type = ExpressionBase<BinaryManagedExpression<L, R, F>>;
  using typename base_type::size_type;
  friend base_type;

  explicit BinaryManagedExpression(const L &lxpr, const R &rxpr, F &&callable = F{}) noexcept : lhs_(lxpr), rhs_(rxpr), op_(std::forward<F>(callable))
  {
    PetscAssertAbort(lhs_.size() == rhs_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Left and right operand size mismatch, %zu != %zu", lhs_.size(), rhs_.size());
  }

  PETSC_NODISCARD size_type size_impl_() const noexcept { return lhs_.size(); };

  PETSC_NODISCARD auto at_impl_(size_type idx) const noexcept
  {
    const auto &lhsv = lhs_.expr_at(idx);

    if (static_cast<const void *>(std::addressof(lhs_)) == static_cast<const void *>(std::addressof(rhs_))) return op_(lhsv, lhsv);
    return op_(lhsv, rhs_.expr_at(idx));
  }

  template <typename... Args>
  PETSC_NODISCARD PetscErrorCode prefetch_impl_(PetscDeviceContext dctx, Args &&...args) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(lhs_.prefetch(dctx, std::forward<Args>(args)...));
    PetscCall(rhs_.prefetch(dctx, std::forward<Args>(args)...));
    PetscFunctionReturn(0);
  }
};

// ==========================================================================================
// EvaluatedManagedExpression
// ==========================================================================================

template <typename T>
class EvaluatedManagedExpression {
public:
  using expression_type = ExpressionBase<T>;
  using size_type       = typename expression_type::size_type;

  #if 0
  template <typename U>
  EvaluatedManagedExpression(U &&expression, PetscDeviceContext dctx) noexcept : expr_(std::forward<U>(expression)), dctx_(dctx)
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, expression.prefetch(dctx));
    PetscFunctionReturnVoid();
  }
  #endif

  PETSC_NODISCARD PetscDeviceContext dctx() const noexcept { return dctx_; }
  PETSC_NODISCARD size_type          size() const noexcept { return expr_.size(); }
  PETSC_NODISCARD auto               expr_at(size_type idx) const noexcept { return expr_.expr_at(idx); }
  PETSC_NODISCARD PetscErrorCode     prefetch() const noexcept { return expr_.prefetch(dctx()); }

private:
  const expression_type &expr_;
  PetscDeviceContext     dctx_;
};

} // namespace expr

template <typename>
class ManagedType;

// ==========================================================================================
// ProxyReference
// ==========================================================================================

template <typename T>
class ProxyReference {
public:
  using value_type   = T;
  using managed_type = ManagedType<value_type>;
  using size_type    = typename managed_type::size_type;

  ProxyReference() noexcept = default;
  ProxyReference(managed_type *, PetscDeviceContext, size_type) noexcept;

  ProxyReference &operator=(const value_type &) noexcept;

  operator value_type() const noexcept;

private:
  managed_type      *man_  = nullptr;
  PetscDeviceContext dctx_ = nullptr;
  size_type          idx_  = 0;
};

template <typename T>
inline ProxyReference<T>::ProxyReference(managed_type *man, PetscDeviceContext dctx, size_type idx) noexcept : man_(man), dctx_(dctx), idx_(idx)
{
}

template <typename T>
inline ProxyReference<T> &ProxyReference<T>::operator=(const value_type &val) noexcept
{
  const auto   fast_assign = man_->is_nosync_available(PETSC_MEMTYPE_HOST);
  PetscMemType mtype;
  value_type  *ptr;

  PetscFunctionBegin;
  PetscAssertAbort(idx_ < man_->size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %zu >= size %zu", idx_, man_->size());
  PetscCallAbort(PETSC_COMM_SELF, man_->get_array_and_memtype(dctx_, PETSC_MEMORY_ACCESS_WRITE, &ptr, &mtype));
  if (fast_assign && PetscMemTypeHost(mtype)) {
    ptr[idx_] = val;
  } else {
    PetscCallAbort(PETSC_COMM_SELF, PetscDeviceArrayCopy(dctx_, std::next(ptr, idx_), std::addressof(val), 1));
  }
  PetscFunctionReturn(*this);
}

template <typename T>
inline ProxyReference<T>::operator value_type() const noexcept
{
  return *man_->host_cbegin(dctx_, true);
}

// ==========================================================================================
// ManagedType
// ==========================================================================================

template <typename T>
class ManagedType : public expr::ExpressionBase<ManagedType<T>> {
  using base_type = expr::ExpressionBase<ManagedType>;
  friend base_type;

public:
  using value_type      = T;
  using pointer         = value_type *;
  using const_pointer   = const value_type *;
  using reference       = value_type &;
  using const_reference = const value_type &;
  using size_type       = std::size_t;
  using storage_type    = memory::managed_storage<value_type>;

  template <typename>
  friend class ManagedType;

  // ==================================================================================== //
  // constructors
  // ==================================================================================== //

  explicit ManagedType() = default;

  ManagedType(PetscDeviceContext, value_type *, value_type *, size_type, PetscCopyMode, PetscCopyMode, PetscOffloadMask) noexcept;
  ManagedType(PetscDeviceContext, size_type) noexcept;
  ~ManagedType() noexcept;

  ManagedType(const ManagedType &) noexcept            = default;
  ManagedType &operator=(const ManagedType &) noexcept = default;

  // steals the other managed types ID
  ManagedType(ManagedType &&) noexcept;
  ManagedType &operator=(ManagedType &&) noexcept;

  template <typename U>
  ManagedType(const expr::EvaluatedManagedExpression<U> &) noexcept;

  // ==================================================================================== //
  // assignment operators
  // ==================================================================================== //

  // TODO
  template <typename U>
  ManagedType &operator=(const expr::EvaluatedManagedExpression<U> &) noexcept = delete;

  // ==================================================================================== //
  // basic getters
  // ==================================================================================== //

  PETSC_NODISCARD size_type        size() const noexcept { return size_; }
  PETSC_NODISCARD size_type        capacity() const noexcept { return std::max(host_.capacity(), device_.capacity()); }
  PETSC_NODISCARD bool             empty() const noexcept { return size() == 0; }
  PETSC_NODISCARD PetscOffloadMask offload_mask() const noexcept { return mask_; }
  PETSC_NODISCARD bool             is_nosync_available(PetscMemType mtype) const noexcept { return PetscMemTypeHost(mtype) ? (pure_ && host_data()) : false; }

  PETSC_NODISCARD ProxyReference<T> at(PetscDeviceContext dctx, size_type idx) noexcept
  {
    PetscFunctionBegin;
    PetscAssertAbort(size() > idx, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index %zu >= size %zu", idx, size());
    PetscFunctionReturn({this, dctx, idx});
  }

  PETSC_NODISCARD value_type at(PetscDeviceContext dctx, size_type idx) const noexcept { return *prefetch_begin_(*this, host_, dctx, true, PETSC_MEMORY_ACCESS_READ); }

  // ==========================================================================================
  // specific getters
  // ==========================================================================================

  PETSC_NODISCARD auto host_data() noexcept { return host_.data(); }
  PETSC_NODISCARD auto host_cdata() const noexcept { return host_.cdata(); }
  PETSC_NODISCARD auto host_data() const noexcept { return host_cdata(); }
  PETSC_NODISCARD auto device_data() noexcept { return device_.data(); }
  PETSC_NODISCARD auto device_cdata() const noexcept { return device_.cdata(); }
  PETSC_NODISCARD auto device_data() const noexcept { return device_cdata(); }

  PETSC_NODISCARD auto data() noexcept { return host_data(); }
  PETSC_NODISCARD auto cdata() const noexcept { return host_data(); }
  PETSC_NODISCARD auto data() const noexcept { return host_data(); }

  // ==========================================================================================
  // Iterators - begin
  // ==========================================================================================

  PETSC_NODISCARD auto host_begin(PetscDeviceContext dctx, bool sync = false, PetscMemoryAccessMode mode = PETSC_MEMORY_ACCESS_READ_WRITE) noexcept { return prefetch_begin_(*this, host_, dctx, sync, mode); }
  PETSC_NODISCARD auto host_cbegin(PetscDeviceContext dctx, bool sync = false) const noexcept { return prefetch_begin_(*this, host_, dctx, sync); }
  PETSC_NODISCARD auto host_begin(PetscDeviceContext dctx, bool sync = false) const noexcept { return host_cbegin(dctx, sync); }

  PETSC_NODISCARD auto device_begin(PetscDeviceContext dctx, bool sync = false, PetscMemoryAccessMode mode = PETSC_MEMORY_ACCESS_READ_WRITE) noexcept { return prefetch_begin_(*this, device_, dctx, sync, mode); }
  PETSC_NODISCARD auto device_cbegin(PetscDeviceContext dctx, bool sync = false) const noexcept { return prefetch_begin_(*this, device_, dctx, sync); }
  PETSC_NODISCARD auto device_begin(PetscDeviceContext dctx, bool sync = false) const noexcept { return device_cbegin(dctx, sync); }

  PETSC_NODISCARD auto begin(PetscDeviceContext dctx = nullptr) noexcept { return host_begin(dctx, true); }
  PETSC_NODISCARD auto cbegin(PetscDeviceContext dctx = nullptr) const noexcept { return host_begin(dctx, true); }
  PETSC_NODISCARD auto begin(PetscDeviceContext dctx = nullptr) const noexcept { return cbegin(dctx); }

  // ==========================================================================================
  // Iterators - end
  // ==========================================================================================

  PETSC_NODISCARD auto host_end(PetscDeviceContext dctx, bool sync = false, PetscMemoryAccessMode mode = PETSC_MEMORY_ACCESS_READ_WRITE) noexcept { return prefetch_end_(*this, host_, dctx, sync, mode); }
  PETSC_NODISCARD auto host_cend(PetscDeviceContext dctx, bool sync = false) const noexcept { return prefetch_end_(*this, host_, dctx, sync); }
  PETSC_NODISCARD auto host_end(PetscDeviceContext dctx, bool sync = false) const noexcept { return host_cend(dctx, sync); }

  PETSC_NODISCARD auto device_end(PetscDeviceContext dctx, bool sync = false, PetscMemoryAccessMode mode = PETSC_MEMORY_ACCESS_READ_WRITE) noexcept { return prefetch_end_(*this, device_, dctx, sync, mode); }
  PETSC_NODISCARD auto device_cend(PetscDeviceContext dctx, bool sync = false) const noexcept { return prefetch_end_(*this, device_, dctx, sync); }
  PETSC_NODISCARD auto device_end(PetscDeviceContext dctx, bool sync = false) const noexcept { return device_cend(dctx, sync); }

  PETSC_NODISCARD auto end(PetscDeviceContext dctx = nullptr) noexcept { return host_end(dctx, true); }
  PETSC_NODISCARD auto cend(PetscDeviceContext dctx = nullptr) const noexcept { return host_end(dctx, true); }
  PETSC_NODISCARD auto end(PetscDeviceContext dctx = nullptr) const noexcept { return cend(dctx); }

  // ==========================================================================================
  // Accessors - front
  // ==========================================================================================

  PETSC_NODISCARD auto front(PetscDeviceContext dctx = nullptr) noexcept { return at(dctx, 0); }
  PETSC_NODISCARD auto front(PetscDeviceContext dctx = nullptr) const noexcept { return cfront(dctx); }
  PETSC_NODISCARD auto cfront(PetscDeviceContext dctx = nullptr) const noexcept { return at(dctx, 0); }

  // ==========================================================================================
  // Accessors - back
  // ==========================================================================================

  PETSC_NODISCARD auto back(PetscDeviceContext dctx = nullptr) noexcept { return at(dctx, size() - 1); }
  PETSC_NODISCARD auto back(PetscDeviceContext dctx = nullptr) const noexcept { return cback(dctx); }
  PETSC_NODISCARD auto cback(PetscDeviceContext dctx = nullptr) const noexcept { return at(dctx, size() - 1); }

  // ==================================================================================== //
  // accessors
  // ==================================================================================== //

  PETSC_NODISCARD PetscErrorCode get_array(PetscDeviceContext, PetscMemType, PetscMemoryAccessMode, PetscBool, value_type **) noexcept;
  PETSC_NODISCARD PetscErrorCode get_array_and_memtype(PetscDeviceContext, PetscMemoryAccessMode, value_type **, PetscMemType * = nullptr) noexcept;
  PETSC_NODISCARD PetscErrorCode clear() noexcept;
  PETSC_NODISCARD PetscErrorCode reserve(PetscDeviceContext, size_type) noexcept;
  template <typename Iterator>
  PETSC_NODISCARD PetscErrorCode assign(PetscDeviceContext, Iterator, Iterator, PetscMemType) noexcept;

private:
  size_type                size_{};
  mutable storage_type     host_{};
  mutable storage_type     device_{};
  mutable PetscOffloadMask mask_{PETSC_OFFLOAD_UNALLOCATED};
  mutable bool             pure_{true};

  PETSC_NODISCARD static storage_type construct_storage_(PetscCopyMode, PetscDeviceContext, PetscMemType, value_type *, size_type) noexcept;

  // clang-format off
  template <typename M, typename S, bool isconst = std::is_const<util::remove_reference_t<M>>::value>
  PETSC_NODISCARD static auto prefetch_begin_(M &&man, S &&storage, PetscDeviceContext dctx, bool sync, PetscMemoryAccessMode mode = PETSC_MEMORY_ACCESS_READ) noexcept -> util::conditional_t<
    isconst,
    typename util::remove_reference_t<S>::const_iterator,
    typename util::remove_reference_t<S>::iterator
  >
  // clang-format on
  {
    static_assert(std::is_same<ManagedType, util::remove_cvref_t<M>>::value, "");
    static_assert(std::is_same<storage_type, util::remove_cvref_t<S>>::value, "");

    PetscFunctionBegin;
    if (isconst) PetscAssertAbort(mode == PETSC_MEMORY_ACCESS_READ, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscMemoryAccessMode must be PETSC_MEMORY_ACCESS_READ when *this is const!");
    PetscCallAbort(PETSC_COMM_SELF, man.prefetch(dctx, storage.mem_type(), mode, sync));
    PetscFunctionReturn(storage.begin());
  }

  // clang-format off
  template <typename M, typename S, bool isconst = std::is_const<util::remove_reference_t<M>>::value>
  PETSC_NODISCARD static auto prefetch_end_(M &&man, S &&storage, PetscDeviceContext dctx, bool sync, PetscMemoryAccessMode mode = PETSC_MEMORY_ACCESS_READ) noexcept -> util::conditional_t<
    isconst,
    typename util::remove_reference_t<S>::const_iterator,
    typename util::remove_reference_t<S>::iterator
  >
  // clang-format on
  {
    static_assert(std::is_same<ManagedType, util::remove_cvref_t<M>>::value, "");
    static_assert(std::is_same<storage_type, util::remove_cvref_t<S>>::value, "");

    PetscFunctionBegin;
    if (isconst) PetscAssertAbort(mode == PETSC_MEMORY_ACCESS_READ, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscMemoryAccessMode must be PETSC_MEMORY_ACCESS_READ when *this is const!");
    PetscCallAbort(PETSC_COMM_SELF, man.prefetch(dctx, storage.mem_type(), mode, sync));
    PetscFunctionReturn(storage.end(man.size()));
  }

  PETSC_NODISCARD PetscErrorCode set_purity_(bool) noexcept;
  PETSC_NODISCARD PetscErrorCode set_offload_mask_(PetscOffloadMask) noexcept;
  PETSC_NODISCARD size_type      size_impl_() const noexcept;
  PETSC_NODISCARD value_type     at_impl_(size_type) const noexcept;
  PETSC_NODISCARD PetscErrorCode prefetch_impl_(PetscDeviceContext, PetscMemType = PETSC_MEMTYPE_HOST, PetscMemoryAccessMode = PETSC_MEMORY_ACCESS_READ, bool = true) const noexcept;

  template <typename Iterator>
  PETSC_NODISCARD PetscErrorCode assign_(PetscDeviceContext, Iterator, Iterator, PetscMemType, int) noexcept;
};

// ==========================================================================================
// ManagedType - Private API
// ==========================================================================================

template <typename T>
inline typename ManagedType<T>::storage_type ManagedType<T>::construct_storage_(PetscCopyMode mode, PetscDeviceContext dctx, PetscMemType mtype, value_type *ptr, size_type n) noexcept
{
  if (ptr) {
    switch (mode) {
    case PETSC_OWN_POINTER:
      return {move_init_t{}, dctx, mtype, ptr, std::next(ptr, n)};
    case PETSC_USE_POINTER:
      return {reference_init_t{}, dctx, mtype, ptr, std::next(ptr, n)};
    case PETSC_COPY_VALUES:
      return {copy_init_t{}, dctx, mtype, ptr, std::next(ptr, n)};
    }
  }
  return {dctx, mtype, n};
}

template <typename T>
inline PetscErrorCode ManagedType<T>::set_purity_(bool purity) noexcept
{
  PetscFunctionBegin;
  pure_ = purity;
  //if (!purity && parent_) parent_->pure_ = purity;
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::set_offload_mask_(PetscOffloadMask mask) noexcept
{
  PetscFunctionBegin;
  if (mask_ != mask) {
    mask_ = mask;
    // should not update the parent if the mask did not change!
    //if (parent_) parent_->mask_ = mask;
  }
  PetscFunctionReturn(0);
}

template <typename T>
inline typename ManagedType<T>::size_type ManagedType<T>::size_impl_() const noexcept
{
  return size();
}

template <typename T>
inline typename ManagedType<T>::value_type ManagedType<T>::at_impl_(size_type idx) const noexcept
{
  return host_cdata()[idx];
}

template <typename T>
inline PetscErrorCode ManagedType<T>::prefetch_impl_(PetscDeviceContext dctx, PetscMemType mtype, PetscMemoryAccessMode mode, bool sync) const noexcept
{
  value_type *dummy;

  PetscFunctionBegin;
  PetscCall(const_cast<ManagedType *>(this)->get_array(dctx, mtype, mode, static_cast<PetscBool>(sync), &dummy));
  PetscFunctionReturn(0);
}

template <typename T>
template <typename Iterator>
inline PetscErrorCode ManagedType<T>::assign_(PetscDeviceContext dctx, Iterator begin, Iterator end, PetscMemType mtype, int fast_assign) noexcept
{
  static_assert(std::is_convertible<typename std::iterator_traits<Iterator>::value_type, value_type>::value, "");
  const auto fast_assign_available = fast_assign > 0;
  const auto on_host               = [&](PetscOffloadMask mask) {
    // we are unallocated or both, copy wherever the source pointer lives
    if (mask == PETSC_OFFLOAD_UNALLOCATED || mask == PETSC_OFFLOAD_BOTH) return PetscMemTypeHost(mtype);
    // we are somewhere, copy to wherever we are
    return PetscOffloadHost(mask);
  }(offload_mask());

  PetscFunctionBegin;
  size_ = std::distance(begin, end);
  if (!size_) PetscFunctionReturn(0);
  // on host? we are pure again. If not we are very much not pure
  PetscCall(set_purity_(on_host && fast_assign_available));
  PetscCall(set_offload_mask_(on_host ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU));
  PetscCall((on_host ? host_ : device_).assign(dctx, std::move(begin), std::move(end), fast_assign_available));
  PetscFunctionReturn(0);
}

// ==========================================================================================
// ManagedType - Public API
// ==========================================================================================

template <typename T>
inline ManagedType<T>::ManagedType(PetscDeviceContext dctx, value_type *host_ptr, value_type *device_ptr, size_type n, PetscCopyMode h_cmode, PetscCopyMode d_cmode, PetscOffloadMask mask) noexcept :
  size_(n), host_(construct_storage_(h_cmode, dctx, PETSC_MEMTYPE_HOST, host_ptr, n)), device_(construct_storage_(d_cmode, dctx, PETSC_MEMTYPE_DEVICE, device_ptr, n)), mask_([=] {
    PetscFunctionBegin;
    if (host_ptr && device_ptr) {
      PetscAssertAbort(mask != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Set both host and device pointer but offloadmask was %s", PetscOffloadMaskToString(mask));
      // this is the only instance in which we believe whatever the user has fed us
      PetscFunctionReturn(mask);
    } else if (host_ptr) {
      // clearly no device_ptr, so must be on cpu
      PetscFunctionReturn(PETSC_OFFLOAD_CPU);
    } else if (device_ptr) {
      // clearly no host_ptr, so must be on gpu
      PetscFunctionReturn(PETSC_OFFLOAD_GPU);
    }
    // user gave us nothing, we are nowhere
    PetscFunctionReturn(PETSC_OFFLOAD_UNALLOCATED);
  }())
{
}

template <typename T>
inline ManagedType<T>::ManagedType(PetscDeviceContext dctx, size_type n) noexcept : ManagedType(dctx, nullptr, nullptr, n, PETSC_OWN_POINTER, PETSC_OWN_POINTER, PETSC_OFFLOAD_UNALLOCATED)
{
}

template <typename T>
inline ManagedType<T>::~ManagedType() noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, host_.destroy());
  PetscCallAbort(PETSC_COMM_SELF, device_.destroy());
  PetscFunctionReturnVoid();
}

// steals the other managed types ID
template <typename T>
inline ManagedType<T>::ManagedType(ManagedType &&other) noexcept :
  base_type(std::move(other)), size_(util::exchange(other.size_, 0)), host_(std::move(other.host_)), device_(std::move(other.device_)), mask_(util::exchange(other.mask_, PETSC_OFFLOAD_UNALLOCATED)), pure_(util::exchange(other.pure_, true))
{
}

template <typename T>
template <typename U>
inline ManagedType<T>::ManagedType(const expr::EvaluatedManagedExpression<U> &expr) noexcept : ManagedType(expr.dctx(), expr.size())
{
  value_type *arr;

  PetscFunctionBegin;
  std::cout << "=== eval constructor\n";
  PetscCallAbort(PETSC_COMM_SELF, get_array(expr.dctx(), PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE, PETSC_TRUE, &arr));
  for (size_type i = 0; i < expr.size(); ++i) arr[i] = expr.expr_at(i);
  PetscFunctionReturnVoid();
}

template <typename T>
inline ManagedType<T> &ManagedType<T>::operator=(ManagedType &&other) noexcept
{
  std::cout << "=== move assignment\n";
  PetscFunctionBegin;
  if (this != &other) {
    base_type::operator=(std::move(other));
    PetscCallAbort(PETSC_COMM_SELF, this->clear());
    size_   = util::exchange(other.size_, 0);
    host_   = std::move(other.host_);
    device_ = std::move(other.device_);
    mask_   = util::exchange(other.mask_, PETSC_OFFLOAD_UNALLOCATED);
    pure_   = util::exchange(other.pure_, true);
  }
  PetscFunctionReturn(*this);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::clear() noexcept
{
  PetscFunctionBegin;
  size_ = 0;
  PetscCall(set_purity_(true));
  PetscCall(set_offload_mask_(PETSC_OFFLOAD_UNALLOCATED));
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::get_array(PetscDeviceContext dctx, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, value_type **ptr) noexcept
{
  const auto get_array_from_storage = [&](storage_type &dest, const storage_type &src, PetscOffloadMask requested_mask) {
    PetscFunctionBegin;
    PetscAssert(requested_mask != PETSC_OFFLOAD_BOTH, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot have %s!", PetscOffloadMaskToString(requested_mask));
    if (PetscUnlikelyDebug(offload_mask() == PETSC_OFFLOAD_UNALLOCATED)) {
      PetscCheck(!dest.data(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "unallocated but dest has data: %p", dest.data());
      PetscCheck(!src.data(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "unallocated but src has data: %p", src.data());
    }

    PetscCall(dest.reserve(dctx, size()));
    if (PetscOffloadUnallocated(offload_mask())) {
      PetscCall(set_offload_mask_(requested_mask));
    } else if (offload_mask() != requested_mask) {
      // if we want any kind of read (read or read_write) and we have valid SRC, we need to copy
      // it now
      if (PetscMemoryAccessRead(mode) && !src.empty() && !PetscOffloadBoth(offload_mask())) {
        PetscCall(set_offload_mask_(PETSC_OFFLOAD_BOTH));
        PetscCall(dest.assign(dctx, src));
      }
      // if we have any kind of write then mask is set to the specific requested version (which
      // must not be OFFLOAD_BOTH)
      if (PetscMemoryAccessWrite(mode)) PetscCall(set_offload_mask_(requested_mask));
    }
    // inform the destination buffer that we intend to modify it per mode
    PetscCall(dest.touch(dctx, mode));
    *ptr = dest.data();
    PetscFunctionReturn(0);
  };

  PetscFunctionBegin;
  std::cout << "get_array(" << (dctx ? PetscObjectCast(dctx)->name : "(unnamed)") << ", " << PetscMemTypeToString(mtype) << ", " << PetscMemoryAccessModeToString(mode) << ", " << PetscBools[sync] << ")\n";
  PetscValidPointer(ptr, 5);
  PetscCheck(!(PetscOffloadUnallocated(offload_mask()) && PetscMemoryAccessRead(mode)), PetscObjectComm(PetscObjectCast(dctx)), PETSC_ERR_ARG_WRONG, "Trying to read (using %s) from a managed type (id %d) that has not been written to (has offload mask %s)", PetscMemoryAccessModeToString(mode), -1, PetscOffloadMaskToString(offload_mask()));
  *ptr = nullptr;
  if (empty()) PetscFunctionReturn(0);

  // retrieve the pointer
  switch (mtype) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(get_array_from_storage(host_, device_, PETSC_OFFLOAD_CPU));
    break;
  case PETSC_MEMTYPE_DEVICE:
    PetscCall(get_array_from_storage(device_, host_, PETSC_OFFLOAD_GPU));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscMemType must be either %s or %s not %d", PetscMemTypeToString(PETSC_MEMTYPE_HOST), PetscMemTypeToString(PETSC_MEMTYPE_DEVICE), static_cast<int>(mtype));
    break;
  }

  // if user intends to write to device in any capacity then we are impure
  if (PetscMemTypeDevice(mtype) && PetscMemoryAccessWrite(mode)) PetscCall(set_purity_(false));
  // REVIEW ME:
  // if we are pure, there is no need to synchronize (I think)
  if (sync && !pure_) {
    PetscCall(PetscDeviceContextSynchronize(dctx));
    if (PetscMemTypeHost(mtype)) PetscCall(set_purity_(true));
  }
  PetscAssert(*ptr, PetscObjectComm(PetscObjectCast(dctx)), PETSC_ERR_PLIB, "ManagedType (id %d) Returned null pointer for mtype %s", -1, PetscMemTypeToString(mtype));
  std::cout << "PetscOffloadMask: " << PetscOffloadMaskToString(offload_mask()) << std::endl;
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::get_array_and_memtype(PetscDeviceContext dctx, PetscMemoryAccessMode mode, value_type **ptr, PetscMemType *mtype) noexcept
{
  PetscMemType retmtype;

  PetscFunctionBegin;
  PetscValidPointer(ptr, 3);
  if (mtype) PetscValidPointer(mtype, 4);
  switch (const auto mask = offload_mask()) {
    // if both prefer CPU, since we may be able to set purity
  case PETSC_OFFLOAD_BOTH:
  case PETSC_OFFLOAD_CPU:
    retmtype = PETSC_MEMTYPE_HOST;
    break;
  case PETSC_OFFLOAD_GPU:
    retmtype = PETSC_MEMTYPE_DEVICE;
    break;
  case PETSC_OFFLOAD_UNALLOCATED: {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    retmtype = dtype == PETSC_DEVICE_HOST ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE;
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support yet for offloadmask %d", mask);
  }
  PetscCall(get_array(dctx, retmtype, mode, PETSC_FALSE, ptr));
  PetscAssert(*ptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, PetscStringize(PetscManagedType) " returned a null pointer for memtype %s as values", PetscMemTypeToString(retmtype));
  if (mtype) *mtype = retmtype;
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::reserve(PetscDeviceContext dctx, size_type n) noexcept
{
  PetscFunctionBegin;
  // TODO reserve only what is possible to reserve!
  PetscCall(host_.reserve(dctx, n));
  PetscCall(device_.reserve(dctx, n));
  PetscFunctionReturn(0);
}

template <typename T>
template <typename Iterator>
inline PetscErrorCode ManagedType<T>::assign(PetscDeviceContext dctx, Iterator begin, Iterator end, PetscMemType mtype) noexcept
{
  PetscFunctionBegin;
  PetscCall(assign_(dctx, std::move(begin), std::move(end), mtype, PetscMemTypeHost(mtype) ? -1 : 0));
  PetscFunctionReturn(0);
}

namespace expr
{

template <typename T>
PETSC_NODISCARD static inline auto eval(PetscDeviceContext dctx, T &&expr) noexcept
{
  return EvaluatedManagedExpression<util::remove_reference_t<T>>{std::forward<T>(expr), dctx};
}

template <typename F, typename L, typename R>
PETSC_NODISCARD static inline auto make_binary_expr(L &&lhs, R &&rhs, F &&fn = F{}) noexcept
{
  return BinaryManagedExpression<util::remove_reference_t<L>, util::remove_reference_t<R>, util::remove_reference_t<F>>(std::forward<L>(lhs), std::forward<R>(rhs), std::forward<F>(fn));
}

template <typename L, typename R>
static inline auto operator*(L &&lhs, R &&rhs) noexcept
{
  return make_binary_expr<std::multiplies<>>(std::forward<L>(lhs), std::forward<R>(rhs));
}

template <typename L, typename R>
static inline auto operator+(L &&lhs, R &&rhs) noexcept
{
  return make_binary_expr<std::plus<>>(std::forward<L>(lhs), std::forward<R>(rhs));
}

template <typename L, typename R>
static inline auto operator-(L &&lhs, R &&rhs) noexcept
{
  return make_binary_expr<std::minus<>>(std::forward<L>(lhs), std::forward<R>(rhs));
}

template <typename L, typename R>
static inline auto operator/(L &&lhs, R &&rhs) noexcept
{
  return make_binary_expr<std::divides<>>(std::forward<L>(lhs), std::forward<R>(rhs));
}

template <typename L, typename R>
static inline auto operator%(L &&lhs, R &&rhs) noexcept
{
  return make_binary_expr<std::modulus<>>(std::forward<L>(lhs), std::forward<R>(rhs));
}

template <typename L, typename R>
static inline auto operator&(L &&lhs, R &&rhs) noexcept
{
  return make_binary_expr<std::bit_and<>>(std::forward<L>(lhs), std::forward<R>(rhs));
}

template <typename L, typename R>
static inline auto operator|(L &&lhs, R &&rhs) noexcept
{
  return make_binary_expr<std::bit_or<>>(std::forward<L>(lhs), std::forward<R>(rhs));
}

template <typename L, typename R>
static inline auto operator^(L &&lhs, R &&rhs) noexcept
{
  return make_binary_expr<std::bit_xor<>>(std::forward<L>(lhs), std::forward<R>(rhs));
}

} // namespace expr

template class ManagedType<PetscReal>;
template class ManagedType<PetscInt>;

using ManagedReal   = ManagedType<PetscReal>;
using ManagedScalar = ManagedType<PetscScalar>;
using ManagedInt    = ManagedType<PetscInt>;

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCMANAGEDTYPE_HPP
