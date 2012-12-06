#include<iostream>

#include<thrust/device_vector.h>
#include<thrust/tuple.h>
#include<thrust/transform.h>
#include<thrust/iterator/zip_iterator.h>

// 3�����x�N�g���i�̃^�v���j
typedef thrust::tuple<double, double, double> Double3;

// 2�����x�N�g���̑傫�����擾����
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

// 3�����x�N�g��Thrust
void Length3Thrust()
{
	// �v�f��
	const int N = 5;

	// x, y, z��������
	double x[N] = {0, 1, 2, 3, 4};
	double y[N] = {1, 2, 3, 4, 5};
	double z[N] = {2, 3, 4, 5, 6};

	// �e�x�N�g���̑傫��
	double length[N];

	// �f�o�C�X�̔z��𐶐�
	thrust::device_vector<double> xVector(x, x + N);
	thrust::device_vector<double> yVector(y, y + N);
	thrust::device_vector<double> zVector(z, z + N);
	thrust::device_vector<double> lengthVector(N);

	// �^�v��������āA���̃^�v���̃C�e���[�^�[���쐬
	auto double3Tuple = thrust::make_tuple(xVector.begin(), yVector.begin(), zVector.begin());
	auto double3Iterator = thrust::make_zip_iterator(double3Tuple);
	
	// �傫�����v�Z
	thrust::transform(double3Iterator, double3Iterator + N,
		lengthVector.begin(),
		getLength3);
	thrust::copy_n(lengthVector.begin(), N, length);

	// ���ʂ�\��
	/*
	* 2.23, 3.74, 5.39, 7.07, 7.77���炢
	*/
	std::cout << "3�����x�N�g��CPU" << std::endl;
	for(int i = 0; i < N; i++)
	{
		std::cout << i << ": " << length[i] << std::endl;
	}
}


// 3�����x�N�g��CPU
void Length3()
{
	// �v�f��
	const int N = 5;

	// x, y, z��������
	double x[N] = {0, 1, 2, 3, 4};
	double y[N] = {1, 2, 3, 4, 5};
	double z[N] = {2, 3, 4, 5, 6};

	// �e�x�N�g���̑傫��
	double length[N];

	// �傫�����v�Z
	for(int i = 0; i < N; i++)
	{
		// ��(x^2 + y^2 + z^2)
		length[i] = std::sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
	}

	// ���ʂ�\��
	/*
	* 2.23, 3.74, 5.39, 7.07, 7.77���炢
	*/
	std::cout << "3�����x�N�g��CPU" << std::endl;
	for(int i = 0; i < N; i++)
	{
		std::cout << i << ": " << length[i] << std::endl;
	}
}

// 2�����x�N�g���̑傫�����擾����
struct GetLength2 : public thrust::binary_function<const double, const double, double>
{
	__host__ __device__
	double operator()(const double& x, const double& y) const
	{
		return std::sqrt(x*x + y*y);
	}
} getLength2;

// 2�����x�N�g��Thrust
void Length2Thrust()
{
	// �v�f��
	const int N = 5;

	// x, y��������
	double x[N] = {0, 1, 2, 3, 4};
	double y[N] = {1, 2, 3, 4, 5};

	// �e�x�N�g���̑傫��
	double length[N];

	// �f�o�C�X�̔z��𐶐�
	thrust::device_vector<double> xVector(x, x + N);
	thrust::device_vector<double> yVector(y, y + N);
	thrust::device_vector<double> lengthVector(N);

	// �傫�����v�Z
	thrust::transform(xVector.begin(), xVector.begin() + N, yVector.begin(), lengthVector.begin(), getLength2);
	thrust::copy_n(lengthVector.begin(), N, length);

	// ���ʂ�\��
	/*
	* 1.00, 2.24, 3.61, 5.00, 6.40���炢
	*/
	std::cout << "2�����x�N�g��Thrust" << std::endl;
	for(int i = 0; i < N; i++)
	{
		std::cout << i << ": " << length[i] << std::endl;
	}
}

// 2�����x�N�g��CPU
void Length2()
{
	// �v�f��
	const int N = 5;

	// x, y��������
	double x[N] = {0, 1, 2, 3, 4};
	double y[N] = {1, 2, 3, 4, 5};

	// �e�x�N�g���̑傫��
	double length[N];

	// �傫�����v�Z
	for(int i = 0; i < N; i++)
	{
		// ��(x^2 + y^2 + z^2)
		length[i] = std::sqrt(x[i]*x[i] + y[i]*y[i]);
	}

	// ���ʂ�\��
	/*
	* 1.00, 2.24, 3.61, 5.00, 6.40���炢
	*/
	std::cout << "2�����x�N�g��CPU" << std::endl;
	for(int i = 0; i < N; i++)
	{
		std::cout << i << ": " << length[i] << std::endl;
	}
}

// �G���g���|�C���g
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