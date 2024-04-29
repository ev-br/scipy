#pragma once

#include <array>
#include "mdspan.h"

namespace scipy {

namespace detail {

/* XXX
 * 1. where to put it? scipy::detail namespace is a placeholder, really.
 * 2. how to avoid duplicating the whole of the layout_stride? Tried
 *    following the suggestion of template<typename LayoutPolicy> struct BoundsCheckingLayoutPolicy
 *    but the syntax defeats me.
 */


struct bounds_checking_layout_stride {
    template <typename Extents>
    class mapping {
      public:
        using extents_type = Extents;
        using index_type = typename extents_type::index_type;
        using size_type = typename extents_type::size_type;
        using rank_type = typename extents_type::rank_type;
        using layout_type = bounds_checking_layout_stride;

      private:
        extents_type m_exts;
        std::array<index_type, extents_type::rank()> m_strides;

      public:
        constexpr mapping() = default;

        constexpr mapping(const Extents &exts, const std::array<index_type, extents_type::rank()> &strides)
            : m_exts(exts), m_strides(strides) {}

        constexpr const extents_type &extents() const noexcept { return m_exts; }

        constexpr const std::array<index_type, extents_type::rank()> &strides() const noexcept { return m_strides; }

        constexpr index_type extent(rank_type i) const noexcept { return m_exts.extent(i); }

        constexpr index_type stride(rank_type i) const noexcept { return m_strides[i]; }

        template <typename... Args>
        constexpr index_type operator()(Args... args) const {
            static_assert(sizeof...(Args) == extents_type::rank(), "index must have same rank as extents");

            index_type indices[extents_type::rank()] = {args...};

            // boundscheck
            for (rank_type i = 0; i < extents_type::rank(); ++i) {
                bool in_bounds = (0 <= indices[i]) && (indices[i] < m_exts.extent(i));
                if(!in_bounds){
                    auto mesg = "OOB: index = " + std::to_string(indices[i]) + " of size = ";
                    mesg = mesg + std::to_string(m_exts.extent(i)) + " in dimension = " + std::to_string(i);
                    throw std::runtime_error(mesg);
                }
            }

            index_type res = 0;
            for (rank_type i = 0; i < extents_type::rank(); ++i) {
                res += indices[i] * m_strides[i];
            }

            return res;
        }
    };
};

} // namespace detail

} // namespace scipy
