#include<iostream>

#include<thrust/device_vector.h>
#include<thrust/tuple.h>
#include<thrust/transform.h>
#include<thrust/iterator/zip_iterator.h>

// 3次元ベクトル（のタプル）
typedef thrust::tuple<double, double, double> Double3;

// 2次元ベクトルの大きさを取得する
struct GetLength3 : public thrust::unary_function<const Double3, double>
{
	__host__ __device__
	double operator()(const Double3& v) const
	{
		double x = v.get<0>();
		double y = v.get<1>();
		double z = v.get<2>();

		return std::sqrt(x*x + y*y + z*z);
	}
} getLength3;

// 3次元ベクトルThrust
void Length3Thrust()
{
	// 要素数
	const int N = 5;

	// x, y, z方向成分
	double x[N] = {0, 1, 2, 3, 4};
	double y[N] = {1, 2, 3, 4, 5};
	double z[N] = {2, 3, 4, 5, 6};

	// 各ベクトルの大きさ
	double length[N];

	// デバイスの配列を生成
	thrust::device_vector<double> xVector(x, x + N);
	thrust::device_vector<double> yVector(y, y + N);
	thrust::device_vector<double> zVector(z, z + N);
	thrust::device_vector<double> lengthVector(N);

	// タプルを作って、そのタプルのイテレーターを作成
	auto double3Tuple = thrust::make_tuple(xVector.begin(), yVector.begin(), zVector.begin());
	auto double3Iterator = thrust::make_zip_iterator(double3Tuple);
	
	// 大きさを計算
	thrust::transform(double3Iterator, double3Iterator + N,
		lengthVector.begin(),
		getLength3);
	thrust::copy_n(lengthVector.begin(), N, length);

	// 結果を表示
	/*
	* 2.23, 3.74, 5.39, 7.07, 7.77ぐらい
	*/
	std::cout << "3次元ベクトルCPU" << std::endl;
	for(int i = 0; i < N; i++)
	{
		std::cout << i << ": " << length[i] << std::endl;
	}
}


// 3次元ベクトルCPU
void Length3()
{
	// 要素数
	const int N = 5;

	// x, y, z方向成分
	double x[N] = {0, 1, 2, 3, 4};
	double y[N] = {1, 2, 3, 4, 5};
	double z[N] = {2, 3, 4, 5, 6};

	// 各ベクトルの大きさ
	double length[N];

	// 大きさを計算
	for(int i = 0; i < N; i++)
	{
		// √(x^2 + y^2 + z^2)
		length[i] = std::sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
	}

	// 結果を表示
	/*
	* 2.23, 3.74, 5.39, 7.07, 7.77ぐらい
	*/
	std::cout << "3次元ベクトルCPU" << std::endl;
	for(int i = 0; i < N; i++)
	{
		std::cout << i << ": " << length[i] << std::endl;
	}
}

// 2次元ベクトルの大きさを取得する
struct GetLength2 : public thrust::binary_function<const double, const double, double>
{
	__host__ __device__
	double operator()(const double& x, const double& y) const
	{
		return std::sqrt(x*x + y*y);
	}
} getLength2;

// 2次元ベクトルThrust
void Length2Thrust()
{
	// 要素数
	const int N = 5;

	// x, y方向成分
	double x[N] = {0, 1, 2, 3, 4};
	double y[N] = {1, 2, 3, 4, 5};

	// 各ベクトルの大きさ
	double length[N];

	// デバイスの配列を生成
	thrust::device_vector<double> xVector(x, x + N);
	thrust::device_vector<double> yVector(y, y + N);
	thrust::device_vector<double> lengthVector(N);

	// 大きさを計算
	thrust::transform(xVector.begin(), xVector.begin() + N, yVector.begin(), lengthVector.begin(), getLength2);
	thrust::copy_n(lengthVector.begin(), N, length);

	// 結果を表示
	/*
	* 1.00, 2.24, 3.61, 5.00, 6.40ぐらい
	*/
	std::cout << "2次元ベクトルThrust" << std::endl;
	for(int i = 0; i < N; i++)
	{
		std::cout << i << ": " << length[i] << std::endl;
	}
}

// 2次元ベクトルCPU
void Length2()
{
	// 要素数
	const int N = 5;

	// x, y方向成分
	double x[N] = {0, 1, 2, 3, 4};
	double y[N] = {1, 2, 3, 4, 5};

	// 各ベクトルの大きさ
	double length[N];

	// 大きさを計算
	for(int i = 0; i < N; i++)
	{
		// √(x^2 + y^2 + z^2)
		length[i] = std::sqrt(x[i]*x[i] + y[i]*y[i]);
	}

	// 結果を表示
	/*
	* 1.00, 2.24, 3.61, 5.00, 6.40ぐらい
	*/
	std::cout << "2次元ベクトルCPU" << std::endl;
	for(int i = 0; i < N; i++)
	{
		std::cout << i << ": " << length[i] << std::endl;
	}
}

// エントリポイント
int main()
{
	// Thrust
	Length2Thrust();
	Length3Thrust();

	// CPU
	Length2();
	Length3();

	return 0;
}