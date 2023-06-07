#ifndef ALL_HEAD
#define ALL_HEAD

#include <iostream>
#include <list>
#include <vector>
#include <queue>
#include <stack>
#include <set>
#include <iterator>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <string>
#include <numeric>
// Definition for a binary tree node.
class TreeNode {
public:
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};

class ListNode {
public:
	int val;
	ListNode* next;
	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}

};

class SubTree
{
public:
	int minVal;
	int maxVal;
	int sumVal;

	SubTree(int mVal, int maVal)
	{
		minVal = mVal;
		maxVal = maVal;
		sumVal = 0;
	}
	SubTree(int mVal, int maVal, int sVal)
	{
		minVal = mVal;
		maxVal = maVal;
		sumVal = sVal;
	}
};

/*c++ using custom compare function!!!!!!
auto c = [](pair<int, pair<int, int>> a, pair<int, pair<int, int>> b)->bool {return a.first < b.first; };
set<pair<int, pair<int, int>>, decltype(c)> map(c);
*/
#endif

