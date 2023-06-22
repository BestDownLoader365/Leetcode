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
#include <array>
#include <stdio.h>
#include <intrin.h>
#include <bitset>
#include <cstdint>
#include <functional>
using namespace std;
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

class SubTree{
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

class UF {
private:
	vector<int> parent;
	vector<int> size;
public:
	UF(int n)
	{
		parent.resize(n);
		size.resize(n);
		for (int i = 0; i < n; i++)
		{
			parent[i] = i;
			size[i] = 1;
		}
	}
	void _union(int p, int q)
	{
		int rootP = find(p);
		int rootQ = find(q);
		if (rootP == rootQ)
			return;
		if (size[p] > size[q])
		{
			parent[rootQ] = rootP;
			size[rootP] += size[rootQ];
		}
		else
		{
			parent[rootP] = rootQ;
			size[rootQ] += size[rootP];
		}
	}
	int find(int x)
	{
		if (parent[x] != x)
			parent[x] = find(parent[x]);
		return parent[x];
	}
	int getMaxConnectSize() 
	{
		int maxSize = 0;
		for (int i = 0; i < (int)parent.size(); i++) 
		{
			if (i == parent[i]) 
				maxSize = max(maxSize, size[i]);
		}
		return maxSize;
	}

};

class Double_Tree_Array {
private:
	const int N = (int)1e5 + 10;
public:
	int lowbit(int x)
	{
		return x & -x;
	}
	void update(int* tr, int x, int v)
	{
		for (int i = x; i < N; i += lowbit(i)) tr[i] += v;
	}
	int query(int* tr, int x)
	{
		int ans = 0;
		for (int i = x; i > 0; i -= lowbit(i)) ans += tr[i];
		return ans;
	}
};

/*c++ using custom compare function!!!!!!
auto c = [](pair<int, pair<int, int>> a, pair<int, pair<int, int>> b)->bool {return a.first < b.first; };
set<pair<int, pair<int, int>>, decltype(c)> map(c);
*/
#endif

