#include "ALL_TEST_In_One.hpp"

int main()
{
	Solution test;
	vector<int> a{1, 2, 2, 1, 1, 0, 2, 2};
	vector<int> b = test.applyOperations(a);
	for (int i : b)
		cout << i << endl;
}