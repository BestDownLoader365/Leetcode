#include <iostream>
using namespace std;

void radixsort(long* seq, long size)
{
	for (long i = 0; i < 2; ++i)
	{
		long* output = new long[size];
		int j = 0;
		int* count = new int[100000];
		int temp = 0;
		for (int k = 0; k < 100000; ++k)
		{
			count[k] = 0;
		}
		for (j = 0; j < size; ++j)
		{
			if (i == 0)
				++count[seq[j] % 100000];
			else
				++count[seq[j] / 100000];
		}
		for (j = 1; j < 100000; ++j)
				count[j] += count[j - 1];
		for (j = size - 1; j >= 0; --j)
		{
			if (i == 0)
			{
				temp = seq[j] % 100000;
				output[count[temp] - 1] = seq[j];
				--count[temp];
			}
			else
			{
				temp = seq[j] / 100000;
				output[count[temp] - 1] = seq[j];
				--count[temp];
			}
		}

		for (j = 0; j < size; ++j)
			seq[j] = output[j];

		delete[] output;
		delete[] count;
	}
}

int main2()
{
	long TC, N, A, B, C, X, Y, V;
	cin >> TC;
	long* output = new long[TC];

	for (long i = 0; i < TC; ++i)
	{
		cin >> N >> A >> B >> C >> X >> Y;
		long* seq = new long[N] ;
		
		seq[0] = A;
		for (long i = 1; i < N; ++i)
		{
			seq[i] = (seq[i - 1] * B + A) % C;
		}
		radixsort(seq, N);

		V = 0;
		for (long j = 0; j < N; ++j)
		{
			V = (V * X + seq[j]) % Y;
		}
		output[i] = V;
	}

	for (long i = 0; i < TC; ++i)
	{
		cout << output[i] << endl;
	}
	return 0;
}