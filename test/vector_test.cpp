#include <boost/test/unit_test.hpp>

#include "cpp-math/vector.hpp"

BOOST_AUTO_TEST_CASE(creation){

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

}

BOOST_AUTO_TEST_CASE(swizzling){

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

BOOST_AUTO_TEST_CASE(comparator_tests){

	// TODO :

}

BOOST_AUTO_TEST_CASE(float1_arithmetic_tests){

	// addition
	{
		float1 a(7.7f);
		float1 b(-7.7f);
		BOOST_CHECK_EQUAL(a+b, float1(0.0f));
		float1 c(114.3f);
		BOOST_CHECK_EQUAL(a+c,float1((float)a + (float)c));
		BOOST_CHECK_EQUAL((float)a+c,(float)a + (float)c);
	}

	// subtraction
	{
		float1 a(3.3f);
		float1 b(8.8f);
		float1 c(-5.5f);
		float1 d(5.5f);
		
		BOOST_CHECK_EQUAL((float)(a+b),3.3f + 8.8f);
		BOOST_CHECK_EQUAL((float)(a-b),3.3f - 8.8f);
		BOOST_CHECK_EQUAL(a + c, a - d);
		BOOST_CHECK_EQUAL(a + d, a - c);
		BOOST_CHECK_EQUAL((float)(c + d), 0.0f);
	}

	// multiplication
	{
		float1 a(3.1f);
		float1 b(8.8f);
		float1 c(-2.5f);
		float1 d(2.5f);
		
		BOOST_CHECK_EQUAL(a*b, ((float)a) * ((float)b));
		BOOST_CHECK_EQUAL(b*c, ((float)b) * ((float)c));
		BOOST_CHECK_EQUAL(b*c, ((float)b) * ((float)c));
		BOOST_CHECK_EQUAL(c*d, ((float)c) * ((float)d));
	}

	// division
	{
		float1 a(3.1f);
		float1 b(8.8f);
		float1 c(-2.5f);
		float1 d(2.5f);
		
		BOOST_CHECK_EQUAL(a/b, ((float)a) / ((float)b));
		BOOST_CHECK_EQUAL(b/c, ((float)b) / ((float)c));
		BOOST_CHECK_EQUAL(b/c, ((float)b) / ((float)c));
		BOOST_CHECK_EQUAL(c/d, ((float)c) / ((float)d));
	}

}

BOOST_AUTO_TEST_CASE(float4_arithmetic_tests){

	// addition
	{
		float x[] = {3.2f, 6.6f, 1.9f};
		float y[] = {-5.2f, 9.4f, 6.7f};
		float z[] = {3.0f, 4.7f, 2.8f};
		float w[] = {7.5f, 0.5f,-4.3f};

		float4 a(x[0], y[0], z[0], w[0]);
		float4 b(x[1], y[1], z[1], w[1]);
		float4 c(x[2], y[2], z[2], w[2]);
		
		BOOST_CHECK_EQUAL((a+b).x(),float1(x[0] + x[1]));
		BOOST_CHECK_EQUAL((a+b).y(),float1(y[0] + y[1]));
		BOOST_CHECK_EQUAL((a+b).z(),float1(z[0] + z[1]));
		BOOST_CHECK_EQUAL((a+b).w(),float1(w[0] + w[1]));

		BOOST_CHECK_EQUAL((b+c).x(),float1(x[1] + x[2]));
		BOOST_CHECK_EQUAL((b+c).y(),float1(y[1] + y[2]));
		BOOST_CHECK_EQUAL((b+c).z(),float1(z[1] + z[2]));
		BOOST_CHECK_EQUAL((b+c).w(),float1(w[1] + w[2]));
	}

	// subtraction
	{
		float x[] = {3.2f, 6.6f, 1.9f};
		float y[] = {-5.2f, 9.4f, 6.7f};
		float z[] = {3.0f, 4.7f, 2.8f};
		float w[] = {7.5f, 0.5f,-4.3f};

		float4 a(x[0], y[0], z[0], w[0]);
		float4 b(x[1], y[1], z[1], w[1]);
		float4 c(x[2], y[2], z[2], w[2]);
		
		BOOST_CHECK_EQUAL((a-b).x(),float1(x[0] - x[1]));
		BOOST_CHECK_EQUAL((a-b).y(),float1(y[0] - y[1]));
		BOOST_CHECK_EQUAL((a-b).z(),float1(z[0] - z[1]));
		BOOST_CHECK_EQUAL((a-b).w(),float1(w[0] - w[1]));

		BOOST_CHECK_EQUAL((b-c).x(),float1(x[1] - x[2]));
		BOOST_CHECK_EQUAL((b-c).y(),float1(y[1] - y[2]));
		BOOST_CHECK_EQUAL((b-c).z(),float1(z[1] - z[2]));
		BOOST_CHECK_EQUAL((b-c).w(),float1(w[1] - w[2]));
	}

	// multiplication
	{
		float x[] = {3.2f, 6.6f, 1.9f};
		float y[] = {-5.2f, 9.4f, 6.7f};
		float z[] = {3.0f, 4.7f, 2.8f};
		float w[] = {7.5f, 0.5f,-4.3f};

		float4 a(x[0], y[0], z[0], w[0]);
		float4 b(x[1], y[1], z[1], w[1]);
		float4 c(x[2], y[2], z[2], w[2]);
		
		BOOST_CHECK_EQUAL((a*b).x(),float1(x[0] * x[1]));
		BOOST_CHECK_EQUAL((a*b).y(),float1(y[0] * y[1]));
		BOOST_CHECK_EQUAL((a*b).z(),float1(z[0] * z[1]));
		BOOST_CHECK_EQUAL((a*b).w(),float1(w[0] * w[1]));

		BOOST_CHECK_EQUAL((b*c).x(),float1(x[1] * x[2]));
		BOOST_CHECK_EQUAL((b*c).y(),float1(y[1] * y[2]));
		BOOST_CHECK_EQUAL((b*c).z(),float1(z[1] * z[2]));
		BOOST_CHECK_EQUAL((b*c).w(),float1(w[1] * w[2]));
	}

	// division
	{
		float x[] = {3.2f, 6.6f, 1.9f};
		float y[] = {-5.2f, 9.4f, 6.7f};
		float z[] = {3.0f, 4.7f, 2.8f};
		float w[] = {7.5f, 0.5f,-4.3f};

		float4 a(x[0], y[0], z[0], w[0]);
		float4 b(x[1], y[1], z[1], w[1]);
		float4 c(x[2], y[2], z[2], w[2]);
		
		BOOST_CHECK_EQUAL((a/b).x(),float1(x[0] / x[1]));
		BOOST_CHECK_EQUAL((a/b).y(),float1(y[0] / y[1]));
		BOOST_CHECK_EQUAL((a/b).z(),float1(z[0] / z[1]));
		BOOST_CHECK_EQUAL((a/b).w(),float1(w[0] / w[1]));

		BOOST_CHECK_EQUAL((b/c).x(),float1(x[1] / x[2]));
		BOOST_CHECK_EQUAL((b/c).y(),float1(y[1] / y[2]));
		BOOST_CHECK_EQUAL((b/c).z(),float1(z[1] / z[2]));
		BOOST_CHECK_EQUAL((b/c).w(),float1(w[1] / w[2]));
	}

}
