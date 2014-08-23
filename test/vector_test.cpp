#include <boost/test/unit_test.hpp>

#include "cpp-math/vector.hpp"

BOOST_AUTO_TEST_CASE(construction){

	{
		const float v = 99.12e5f;
		const float1 v1(v);
		BOOST_CHECK_EQUAL((float)v1, v);
	}

	{

		float x = 6.7f;
		float y = -0.0f;
		float z = -5.2f;
		float w = 7.8f;
		float4 vector(x, y, z, w);
		BOOST_CHECK_EQUAL((float)vector.x(), x);
		BOOST_CHECK_EQUAL((float)vector.y(), y);
		BOOST_CHECK_EQUAL((float)vector.z(), z);
		BOOST_CHECK_EQUAL((float)vector.w(), w);

		
		float4 vector_with_zero(x,y,z);
		BOOST_CHECK_EQUAL((float)vector_with_zero.x(), x);
		BOOST_CHECK_EQUAL((float)vector_with_zero.y(), y);
		BOOST_CHECK_EQUAL((float)vector_with_zero.z(), z);
		BOOST_CHECK_EQUAL((float)vector_with_zero.w(), 0.0f);

	}

	{
		const float1 a(5.7f);
		const float1 b(7.8f);
		const float1 c(5.7f);
		const float1 d(std::numeric_limits<float>::quiet_NaN());


		BOOST_CHECK_EQUAL(a, c);
		BOOST_CHECK_NE(a, b);
		float1 x = b;
		BOOST_CHECK_EQUAL(x, b);
		
		BOOST_CHECK_NE(d, c);
		
		BOOST_CHECK_EQUAL(a, a);
		BOOST_CHECK_EQUAL(b, b);
		BOOST_CHECK_EQUAL(c, c);
		BOOST_CHECK_NE(d, d);

	}

	{
		// float1 is likely to respond weirdly
		// when the insignificant components are different 
		// to the significant one
		// ==== Test how robust this is
		__m128 a = _mm_set1_ps(5.0f);
		__m128 b = _mm_set1_ps(120.0f);

		__m128 c = _mm_move_ss(a,b);
		// c is now
		// (120,5,5,5)
		
		float1 vector_a(a);
		float1 vector_b(b);
		float1 vector_c(c);

		BOOST_CHECK_NE(vector_a,vector_b);
		BOOST_CHECK_EQUAL(vector_c,vector_c);
		BOOST_CHECK_NE(vector_a,vector_c);
		BOOST_CHECK_EQUAL(vector_b,vector_c);

	}

	{
		
		float x = 6.7f;
		float y = 7.8f;
		float z = -5.2f;
		float w = -0.0f;
		
		float4 xyzw(x, y, z, w);
		float4 wzyx(swizzle<0,1,2,3>(xyzw));
		float4 xwzy(swizzle<3,0,1,2>(xyzw));
		
		BOOST_CHECK_EQUAL((float)xyzw.x(),x);
		BOOST_CHECK_EQUAL((float)xyzw.y(),y);
		BOOST_CHECK_EQUAL((float)xyzw.z(),z);
		BOOST_CHECK_EQUAL((float)xyzw.w(),w);

		BOOST_CHECK_EQUAL((float)wzyx.x(),w);
		BOOST_CHECK_EQUAL((float)wzyx.y(),z);
		BOOST_CHECK_EQUAL((float)wzyx.z(),y);
		BOOST_CHECK_EQUAL((float)wzyx.w(),x);
		
		BOOST_CHECK_EQUAL((float)xwzy.x(),x);
		BOOST_CHECK_EQUAL((float)xwzy.y(),w);
		BOOST_CHECK_EQUAL((float)xwzy.z(),z);
		BOOST_CHECK_EQUAL((float)xwzy.w(),y);
	}

}