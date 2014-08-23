#pragma once
#include "cpp-math/detail/config.hpp"
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>


BOOST_ALIGNMENT(16) union __f128 {
	struct {
		float w, z, y, x;
	} floats;
	__m128 reg;
};

/*
	The template arguments select which of the
	source elements to transfer to the dest.

	Example from MSDN
	   127							   0
	m1 =|  a   |   b   |   c   |   d   |
	m2 =|  e   |   f   |   g   |   h   |

	m3 = swizzle<1,0,3,2>( m1, m2)
	   =|  g   |   h   |   a   |   b   |
*/
template<int X, int Y, int Z, int W>
__m128 CPPMATH_INLINE swizzle(const __m128& a, const __m128& b){
	return _mm_shuffle_ps(a, b, _MM_SHUFFLE(X, Y, Z, W));
}

template<int X, int Y, int Z, int W>
__m128 CPPMATH_INLINE swizzle(const __m128& a){
	return swizzle< X, Y, Z, W >(a, a);
}




class float1 {
public:
	CPPMATH_INLINE explicit float1(const float& v)
	{
		_data.reg = _mm_set_ss(v);
	}

	CPPMATH_INLINE explicit float1(const __m128& v)
	{
		_data.reg = v;
	}

	CPPMATH_INLINE operator float() const {
		return _data.floats.w;
	}
	
	CPPMATH_INLINE operator __m128() const {
		return _data.reg;
	}
	
	CPPMATH_INLINE bool operator != (const float1& other){
		__m128i vcmp = _mm_castps_si128(_mm_cmpneq_ps(_data.reg,(__m128)other));
		int test = _mm_movemask_epi8(vcmp);
		return (test & 0xF) != 0;
	}
	
	CPPMATH_INLINE bool operator == (const float1& other){
		__m128i vcmp = _mm_castps_si128(_mm_cmpeq_ps(_data.reg,(__m128)other));
		int test = _mm_movemask_epi8(vcmp);
		return (test & 0x000F) != 0;
	}

private:
	__f128 _data;
};

class float4 {
public:
	CPPMATH_INLINE float4(const float& x, const float& y, const float& z, const float& w = 0.0f)
	{
		_data.reg = _mm_set_ps(x,y,z,w);
	}

	CPPMATH_INLINE explicit float4(const __m128& v){
		_data.reg = v;
	}

	CPPMATH_INLINE operator __m128 () const {
		return _data.reg;
	}

	// these accessors should very definitely use swizzles
	CPPMATH_INLINE float1 x() const {
		return float1(swizzle<3,3,3,3>(_data.reg,_data.reg));
	}

	CPPMATH_INLINE float1 y() const {
		return float1(swizzle<2,2,2,2>(_data.reg,_data.reg));
	}

	CPPMATH_INLINE float1 z() const {
		return float1(swizzle<1,1,1,1>(_data.reg,_data.reg));
	}

	CPPMATH_INLINE float1 w() const {
		return float1(swizzle<0,0,0,0>(_data.reg,_data.reg));
	}

	CPPMATH_INLINE bool operator != (const float4& other){
		__m128i vcmp = _mm_castps_si128(_mm_cmpneq_ps(_data.reg,(__m128)other));
		uint16_t test = _mm_movemask_epi8(vcmp);
		return test != 0x0;
	}
	
	CPPMATH_INLINE bool operator == (const float4& other){
		__m128i vcmp = _mm_castps_si128(_mm_cmpeq_ps(_data.reg,(__m128)other));
		uint16_t test = _mm_movemask_epi8(vcmp);
		return test == std::numeric_limits<uint16_t>::max();
	}

private:

	__f128 _data;
};

