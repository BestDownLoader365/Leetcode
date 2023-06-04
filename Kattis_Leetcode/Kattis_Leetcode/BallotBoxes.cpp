#include <iostream>
#include <queue>
#include <cmath>
using namespace std;

long findanswer(long* people,long cities,long boxes)
{
	long left = 0;
	long right = 5000000;
	while (left <= right)
	{
		long mid = (left + right) / 2;
		long current_box = 0;
		for (long i = 0; i < cities; ++i)
			current_box += (long)ceil((double)people[i]/mid);

		if (current_box <= boxes)
			right = mid - 1;
		else
			left = mid + 1;
	}
	return left;
}

int main1()
{
	long cities, boxes;
	cin >> cities;
	cin >> boxes;

	queue<long> output;
	while (!(cities == -1 && boxes == -1))
	{
		long* people = new long[cities];
		long current;
		for (long i = 0; i < cities; ++i)
		{
			cin >> current;
			people[i] = current;
		}

		long answer = findanswer(people,cities,boxes);
		output.push(answer);

		cin >> cities;
		cin >> boxes;
		delete[] people;
	}

	while (output.size() > 0)
	{
		cout << output.front() << endl;
		output.pop();
	}

	return 0;
}