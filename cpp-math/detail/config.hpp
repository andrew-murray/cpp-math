#pragma once
#include <boost/config.hpp>

#ifdef _DEBUG
#    define CPPMATH_INLINE inline
#else
#ifdef NDEBUG
#if   defined(_MSC_VER)
#    define CPPMATH_INLINE __forceinline
#elif defined(__GNUC__) && __GNUC__ > 3
#    define CPPMATH_INLINE inline __attribute__ ((always_inline))
#else
#    define CPPMATH_INLINE inline
#endif
#else
#    define CPPMATH_INLINE inline
#endif
#endif