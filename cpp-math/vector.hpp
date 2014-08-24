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




class float1{
public:
	typedef float1 component_type;
	typedef float value_type;

	class float1comparator{
	public:
		CPPMATH_INLINE explicit float1comparator(const __m128& v){
			_data.reg = v;
		}

		CPPMATH_INLINE operator __m128 () const {
			return _data.reg;
		}
		
		CPPMATH_INLINE float1comparator operator == (const float1comparator& other) const {
			return float1comparator(_mm_cmpeq_ss(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float1comparator operator != (const float1comparator& other) const {
			return float1comparator(_mm_cmpneq_ss(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float1comparator operator > (const float1comparator& other) const {
			return float1comparator(_mm_cmpgt_ss(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float1comparator operator >= (const float1comparator& other) const {
			return float1comparator(_mm_cmpge_ss(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float1comparator operator < (const float1comparator& other) const {
			return float1comparator(_mm_cmplt_ss(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float1comparator operator <= (const float1comparator& other) const {
			return float1comparator(_mm_cmple_ss(_data.reg,(__m128)other));
		}
		
		static CPPMATH_INLINE uint16_t mask(const float1comparator& input){
			__m128i input_as_intvec = _mm_castps_si128((__m128)input);
			uint16_t returned_mask = _mm_movemask_epi8(input_as_intvec);
			return returned_mask;
		}

	private:
		__f128 _data;
	};

	CPPMATH_INLINE explicit float1(const float& v){
		_data.reg = _mm_set_ss(v);
	}

	CPPMATH_INLINE explicit float1(const __m128& v){
		_data.reg = v;
	}

	CPPMATH_INLINE operator float() const {
		return _data.floats.w;
	}
	
	CPPMATH_INLINE operator __m128() const {
		return _data.reg;
	}
	
	CPPMATH_INLINE bool operator != (const float1& other) const {
		const float1comparator us((__m128)*this);
		const float1comparator them((__m128)other);
		int test = float1comparator::mask(us != them);
		return (test & 0xF) != 0;
	}
	
	CPPMATH_INLINE bool operator == (const float1& other) const {
		const float1comparator us((__m128)*this);
		const float1comparator them((__m128)other);
		int test = float1comparator::mask(us == them);
		return (test & 0xF) != 0;
	}
	
	CPPMATH_INLINE float1 operator & (const float1& other) const {
		return float1(_mm_and_ps(_data.reg,(__m128)other));
	}
		
	CPPMATH_INLINE float1 operator | (const float1& other) const {
		return float1(_mm_or_ps(_data.reg,(__m128)other));
	}
		
	CPPMATH_INLINE float1 operator ^ (const float1& other) const {
		return float1(_mm_xor_ps(_data.reg,(__m128)other));
	}

	
	CPPMATH_INLINE float1 operator + (const float1& other){
		return float1(_mm_add_ss(_data.reg,(__m128)other));
	}
	
	CPPMATH_INLINE float1 operator - (const float1& other){
		return float1(_mm_sub_ss(_data.reg,(__m128)other));
	}
	
	CPPMATH_INLINE float1 operator * (const float1& other){
		return float1(_mm_mul_ss(_data.reg,(__m128)other));
	}
	
	CPPMATH_INLINE float1 operator / (const float1& other){
		return float1(_mm_div_ss(_data.reg,(__m128)other));
	}

private:
	__f128 _data;
};

class float4{
public:
	typedef float1 component_type;
	typedef float value_type;

	class float4comparator{
	public:
		CPPMATH_INLINE explicit float4comparator(const __m128& v){
			_data.reg = v;
		}

		CPPMATH_INLINE operator __m128 () const {
			return _data.reg;
		}
		
		CPPMATH_INLINE float4comparator operator == (const float4comparator& other) const {
			return float4comparator(_mm_cmpeq_ps(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float4comparator operator != (const float4comparator& other) const {
			return float4comparator(_mm_cmpneq_ps(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float4comparator operator > (const float4comparator& other) const {
			return float4comparator(_mm_cmpgt_ps(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float4comparator operator >= (const float4comparator& other) const {
			return float4comparator(_mm_cmpge_ps(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float4comparator operator < (const float4comparator& other) const {
			return float4comparator(_mm_cmplt_ps(_data.reg,(__m128)other));
		}
		
		CPPMATH_INLINE float4comparator operator <= (const float4comparator& other) const {
			return float4comparator(_mm_cmple_ps(_data.reg,(__m128)other));
		}
		
		static CPPMATH_INLINE uint16_t mask(const float4comparator& input){
			__m128i input_as_intvec = _mm_castps_si128((__m128)input);
			uint16_t returned_mask = _mm_movemask_epi8(input_as_intvec);
			return returned_mask;
		}

	private:
		__f128 _data;
	};


	CPPMATH_INLINE float4(const float& x, const float& y, const float& z, const float& w = 0.0f){
		_data.reg = _mm_set_ps(x,y,z,w);
	}

	CPPMATH_INLINE explicit float4(const __m128& v){
		_data.reg = v;
	}

	CPPMATH_INLINE operator __m128 () const {
		return _data.reg;
	}

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
		const float4comparator us((__m128)*this);
		const float4comparator them((__m128)other);
		const uint16_t test = float4comparator::mask(us!=them);
		return test != 0x0;
	}
	
	CPPMATH_INLINE bool operator == (const float4& other){
		const float4comparator us((__m128)*this);
		const float4comparator them((__m128)other);
		const uint16_t test = float4comparator::mask(us==them);
		return test == std::numeric_limits<uint16_t>::max();
	}

	CPPMATH_INLINE float4 operator & (const float4& other){
		return float4(_mm_and_ps(_data.reg,(__m128)other));
	}
		
	CPPMATH_INLINE float4 operator | (const float4& other){
		return float4(_mm_or_ps(_data.reg,(__m128)other));
	}
		
	CPPMATH_INLINE float4 operator ^ (const float4& other){
		return float4(_mm_xor_ps(_data.reg,(__m128)other));
	}

	CPPMATH_INLINE float4 operator + (const float4& other){
		return float4(_mm_add_ps(_data.reg,(__m128)other));
	}
	
	CPPMATH_INLINE float4 operator - (const float4& other){
		return float4(_mm_sub_ps(_data.reg,(__m128)other));
	}
	
	CPPMATH_INLINE float4 operator * (const float4& other){
		return float4(_mm_mul_ps(_data.reg,(__m128)other));
	}
	
	CPPMATH_INLINE float4 operator / (const float4& other){
		return float4(_mm_div_ps(_data.reg,(__m128)other));
	}

private:
	__f128 _data;
};
