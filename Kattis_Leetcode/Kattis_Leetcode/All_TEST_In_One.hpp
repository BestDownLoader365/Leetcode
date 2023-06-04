#ifndef ALL_TEST
#define ALL_TEST

#include "All_Header.hpp"
#include <iostream>
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
using namespace std;
class Solution {

public:
	//TEST_316 start
	string removeDuplicateLetters(string s) 
	{
		vector<int> stock(26);
		vector<bool> visited(26);
		string ans = "";
		for (char ch : s)
			stock[ch - 'a']++;

		for (char ch : s)
		{
			if (!visited[ch - 'a'])
			{
				visited[ch - 'a'] = true;
				while (!(ans.empty()) && (ans.back() > ch) && (stock[ans.back() - 'a'] > 0))
				{
					visited[ans.back() - 'a'] = false;
					ans.pop_back();
				}
				ans.push_back(ch);
			}
			stock[ch - 'a']--;
			cout << ans << endl;
		}

		return ans;
	}

	//TEST_316 end 

	//TEST_2465 start
	int distinctAverages(vector<int>& nums) 
	{
		unordered_set<double> stock;
		sort(nums.begin(), nums.end());

		int p1 = 0;
		int p2 = (int)nums.size() - 1;
		while (p1 < p2)
		{
			double curr = (nums[p1] + nums[p2]) / (double)2;
			stock.insert(curr);
			p1++;
			p2--;
		}
		if (p1 == p2)
			stock.insert(nums[p1]);
		return (int)stock.size();
	}
	//TEST_2465 end

	//TEST_134 start
	//暴力法失败了，寄，这个是抄的
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost) 
	{
		int curSum = 0;
		int totalSum = 0;
		int start = 0;
		for (int i = 0; i < (int)gas.size(); i++) {
			curSum += gas[i] - cost[i];
			totalSum += gas[i] - cost[i];
			if (curSum < 0) {   // 当前累加rest[i]和 curSum一旦小于0
				start = i + 1;  // 起始位置更新为i+1
				curSum = 0;     // curSum从0开始
			}
		}
		if (totalSum < 0) return -1; // 说明怎么走都不可能跑一圈了
		return start;
	}
	//TEST_134 end

	//TEST_45 start
	//简易版
	int jump_easy(vector<int>& nums)
	{
		int max_far = 0;// 目前能跳到的最远位置
		int step = 0;   // 跳跃次数
		int end = 0;    // 上次跳跃可达范围右边界（下次的最右起跳点）
		for (int i = 0; i < nums.size() - 1; i++)
		{
			max_far = std::max(max_far, i + nums[i]);
			// 到达上次跳跃能到达的右边界了
			if (i == end)
			{
				end = max_far;  // 目前能跳到的最远位置变成了下次起跳位置的边界
				step++;         // 进入下一次跳跃
			}
		}
		return step;
	}
	//复杂版
	int jump(vector<int>& nums) 
	{
		int n = (int)nums.size();
		vector<int> dp(n);
		for (int i = 1; i < n; i++)
			dp[i] = INT_MAX;
		dp[0] = 0;
		for (int i = 0; i < n; i++)
		{
			int j = i;
			while (nums[i] > 0)
			{
				j++;
				if (j >= n)
					break;
				dp[j] = min(dp[j], dp[i] + 1);
				nums[i]--;
			}
		}
		return dp[n - 1];
	}
	//TEST_45 end

	//TEST_55 start
	//这个是简化版
	bool canJump_easy(vector<int>& nums) {
		int n = (int)nums.size();
		int rightmost = 0;
		for (int i = 0; i < n; ++i) {
			if (i <= rightmost) {
				rightmost = max(rightmost, i + nums[i]);
				if (rightmost >= n - 1) {
					return true;
				}
			}
		}
		return false;
	}
	//这个是复杂了，不需要记录每一个节点是否可以被访问，只需要维护最右边可以访问的节点
	bool canJump(vector<int>& nums) 
	{
		int n = (int)nums.size();
		vector<bool> can_visit(n);
		can_visit[0] = true;
		for (int i = 0; i < n; i++)
		{
			int j = i;
			if (can_visit[i])
			{
				while (nums[i] > 0)
				{
					j++;
					if (j >= n)
						break;
					if (j == n - 1)
						return true;
					can_visit[j] = true;
					nums[i]--;
				}
			}
		}
		return can_visit[n - 1];
	}
	//TEST_55 end

	//TEST_1156 start
	//不会做，寄
	int maxRepOpt1(string text) 
	{
		unordered_map<char, int> count;
		for (auto c : text) {
			count[c]++;
		}

		int res = 0;
		for (int i = 0; i < text.size(); ) {
			// step1: 找出当前连续的一段 [i, j)
			int j = i;
			while (j < text.size() && text[j] == text[i]) {
				j++;
			}
			int cur_cnt = j - i;

			// step2: 如果这一段长度小于该字符出现的总数，并且前面或后面有空位，则使用 cur_cnt + 1 更新答案
			if (cur_cnt < count[text[i]] && (j < text.size() || i > 0)) {
				res = max(res, cur_cnt + 1);
			}

			// step3: 找到这一段后面与之相隔一个不同字符的另一段 [j + 1, k)，如果不存在则 k = j + 1 
			int k = j + 1;
			while (k < text.size() && text[k] == text[i]) {
				k++;
			}
			//这里要min(k - i, count[text[i]])是为了防止后面被替换过来的字符a就是k位的字符
			res = max(res, min(k - i, count[text[i]]));
			i = j;
		}
		return res;
	}
	//TEST_1156 end

	//TEST_2007 start
	vector<int> findOriginalArray(vector<int>& changed)
	{
		if ((changed.size() & 1) == 1)
			return {};
		sort(changed.begin(), changed.end());
		vector<int> ans;
		map<int, int> stock;
		for (int i : changed)
			stock[i]++;
		int cnt = 0;
		for (size_t i = 0; i < changed.size(); i++)
		{
			if (stock[changed[i]] > 0)
			{
				if (stock[changed[i] * 2] > 0)
				{
					stock[changed[i]]--;
					stock[changed[i] * 2]--;
					ans.emplace_back(changed[i]);
				}
				else
					return {};
			}
		}
		return ans;
	}
	//TEST_2007 end

	//TEST_984 start
	string strWithout3a3b(int A, int B) 
	{
		if (B == 0) return string(A, 'a');
		if (A == 0) return string(B, 'b');
		if (A == B) return "ab" + strWithout3a3b(A - 1, B - 1);
		return A > B ? "aab" + strWithout3a3b(A - 2, B - 1) : "bba" + strWithout3a3b(A - 1, B - 2);
	}
	//TEST_984 end

	//TEST_42 start
	//动态规划做法
	int trap(vector<int>& height)
	{
		int ans = 0;
		size_t n = height.size();
		vector<int> leftMax(n);
		vector<int> rightMax(n);
		leftMax[0] = height[0];
		rightMax[n - 1] = height[n - 1];
		for (size_t i = 1; i < n; i++)
		{
			leftMax[i] = max(leftMax[i - 1], height[i]);
			rightMax[n - 1 - i] = max(rightMax[n - i], height[n - i - 1]);
		}
		for (size_t i = 0; i < n; i++)
		{
			ans += min(leftMax[i], rightMax[i]) - height[i];
		}
		return ans;
	}
	//TEST_42 end

	//TEST_2559 start
	vector<int> vowelStrings(vector<string>& words, vector<vector<int>>& queries)
	{
		vector<int> ans(queries.size());
		vector<int> prefix(words.size() + 1);
		prefix[0] = 0;
		vector<char> vowel{'a', 'e', 'i', 'o', 'u'};
		for (size_t i = 0; i < words.size(); i++)
		{
			bool start = false;
			bool end = false;
			for (size_t j = 0; j < 5; j++)
			{
				if (words[i][0] == vowel[j])
					start = true;
				if (words[i][words[i].length() - 1] == vowel[j])
					end = true;
			}
			if (start && end)
				prefix[i + 1] = prefix[i] + 1;
			else
				prefix[i + 1] = prefix[i];
		}

		for (size_t i = 0; i < queries.size(); i++)
		{
			ans[i] = prefix[queries[i][1] + 1] - prefix[queries[i][0]];
		}
		return ans;
	}
	//TEST_2559 end

	//TEST_628 start
	int maximumProduct_2(vector<int>& nums) {
		// 最小的和第二小的
		int min1 = INT_MAX, min2 = INT_MAX;
		// 最大的、第二大的和第三大的
		int max1 = INT_MIN, max2 = INT_MIN, max3 = INT_MIN;

		for (int x : nums) {
			if (x < min1) {
				min2 = min1;
				min1 = x;
			}
			else if (x < min2) {
				min2 = x;
			}

			if (x > max1) {
				max3 = max2;
				max2 = max1;
				max1 = x;
			}
			else if (x > max2) {
				max3 = max2;
				max2 = x;
			}
			else if (x > max3) {
				max3 = x;
			}
		}

		return max(min1 * min2 * max1, max1 * max2 * max3);
	}
	//排序解法
	int maximumProduct(vector<int>& nums) {
		sort(nums.begin(), nums.end());
		int n = (int)nums.size();
		return max(nums[0] * nums[1] * nums[n - 1], nums[n - 3] * nums[n - 2] * nums[n - 1]);
	}
	//TEST_628 end

	//TEST_414 start
	int thirdMax(vector<int>& nums) 
	{
		int* a = nullptr, * b = nullptr, * c = nullptr;
		for (int& num : nums) {
			if (a == nullptr || num > *a) {
				c = b;
				b = a;
				a = &num;
			}
			else if (*a > num && (b == nullptr || num > *b)) {
				c = b;
				b = &num;
			}
			else if (b != nullptr && *b > num && (c == nullptr || num > *c)) {
				c = &num;
			}
		}
		return c == nullptr ? *a : *c;
	}
	//TEST_414 end

	//TEST_2517 start
	int maximumTastiness(vector<int>& price, int k) {
		int n = (int)price.size();
		sort(price.begin(), price.end());
		int left = 0, right = price[n - 1] - price[0];
		while (left < right) {
			int mid = (left + right + 1) >> 1;
			if (check(price, k, mid)) {
				left = mid;
			}
			else {
				right = mid - 1;
			}
		}
		return left;
	}

	bool check(const vector<int>& price, int k, int tastiness) {
		int prev = INT_MIN >> 1;
		int cnt = 0;
		for (int p : price) {
			if (p - prev >= tastiness) {
				cnt++;
				prev = p;
			}
		}
		return cnt >= k;
	}
	//TEST_2517 end

	//TEST_16 start
	int threeSumClosest(vector<int>& nums, int target) 
	{
		int diff = INT_MAX;
		int ans = INT_MIN;
		int n = (int)nums.size();
		sort(nums.begin(), nums.end());
		for(int i = 0; i < n; i++)
		{
			int left = target - nums[i];
			int p2 = 0;
			int p3 = n - 1;
			while (p2 < p3)
			{
				if (p2 == i)
				{
					p2++;
					continue;
				}
				if (p3 == i)
				{
					p3--;
					continue;
				}
				int curr = nums[p2] + nums[p3] - left;
				if (abs(curr) < diff)
				{
					diff = abs(curr);
					ans = nums[p2] + nums[p3] + nums[i];
				}
				if (curr > 0)
					p3--;
				else
					p2++;
			}
		}
		return ans;
	}
	//TEST_16 end

	//TEST_47 start
	void DFS_47(vector<int>& combine, vector<vector<int>>& ans, vector<bool>& pass, size_t n, vector<int> nums)
	{
		if (combine.size() == n)
		{
			ans.push_back(combine);
			return;
		}
		for (int i = 0; i < n; i++)
		{
			if (!pass[i])
			{
				pass[i] = true;
				combine.push_back(nums[i]);
				DFS_47(combine, ans, pass, n, nums);
				combine.pop_back();
				pass[i] = false;
			}

		}
	}
	vector<vector<int>> permuteUnique(vector<int>& nums) 
	{
		sort(nums.begin(), nums.end());
		vector<int> combine;
		vector<vector<int>> ans;
		size_t n = nums.size();
		vector<bool> pass(n);
		DFS_47(combine, ans, pass, n, nums);
		return ans;
	}
	//TEST_47 end

	//TEST_46 start
	void DFS_46(vector<int>& combine, vector<vector<int>>& ans, vector<bool>& pass, size_t n, vector<int> nums)
	{
		if (combine.size() == n)
		{
			ans.push_back(combine);
			return;
		}
		for (int i = 0; i < n; i++)
		{
			if (i > 0 && nums[i] == nums[i - 1])
				continue;
			if (!pass[i])
			{
				pass[i] = true;
				combine.push_back(nums[i]);
				DFS_46(combine, ans, pass, n, nums);
				combine.pop_back();
				pass[i] = false;
			}

		}
	}
	vector<vector<int>> permute(vector<int>& nums) 
	{
		vector<int> combine;
		vector<vector<int>> ans;

		size_t n = nums.size();
		vector<bool> pass(n);
		DFS_46(combine, ans, pass, n, nums);
		return ans;
	}
	//TEST_46 end

	//TEST_1110 start
	void DFS_1110(vector<TreeNode*>& ans, set<int>& stock, TreeNode*& node)
	{
		if (!node)
			return;
		DFS_1110(ans, stock, node->left);
		DFS_1110(ans, stock, node->right);

		if (stock.find(node->val) != stock.end())
		{
			if (node->left)
				ans.emplace_back(node->left);
			if (node->right)
				ans.emplace_back(node->right);
			node = nullptr;
		}

	}
	vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete)
	{
		vector<TreeNode*> ans;
		set<int> stock;
		for (int i : to_delete)
			stock.insert(i);
		DFS_1110(ans, stock, root);
		if (root)
			ans.push_back(root);
		return ans;
	}
	//TEST_1110 end

	//TEST_1130 start
	//动态规划，容易想到
	int mctFromLeafValues(vector<int>& arr) {
		int n = (int)arr.size();
		vector<vector<int>> dp(n, vector<int>(n, INT_MAX / 4)), mval(n, vector<int>(n));
		for (int i = 0; i < n; i++)
		{
			mval[i][i] = arr[i];
			for (int j = i + 1; j < n; j++)
				mval[i][j] = max(mval[i][j - 1], arr[j]);
		}
		for (int j = 0; j < n; j++) {
			dp[j][j] = 0;
			for (int i = j - 1; i >= 0; i--) {
				for (int k = i; k < j; k++) {
					dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + mval[i][k] * mval[k + 1][j]);
				}
			}
		}
		return dp[0][n - 1];
	}
	//TEST_1130 end

	//TEST_59 start
	vector<vector<int>> generateMatrix(int n) 
	{
		int element = n * n;
		vector<vector<int>> ans(n, vector<int>(n));
		vector<pair<int, int>> direc{{ 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }};
		vector<vector<bool>> visited(n, vector<bool>(n));
		int x = 0;
		int y = 0;
		int dir = 0;
		for (int i = 0; i < element; i++)
		{
			ans[x][y] = i + 1;
			visited[x][y] = true;
			int new_x = x + direc[dir].first;
			int new_y = y + direc[dir].second;
			if ((new_x == 0 && new_y == n) || (new_x == n && new_y == n - 1) || (new_x == n - 1 && new_y == -1) || visited[new_x][new_y])
			{
				dir++;
				if (dir == 4)
					dir = 0;
				new_x = x + direc[dir].first;
				new_y = y + direc[dir].second;
			}
			x = new_x;
			y = new_y;
		}
		return ans;
	}
	//TEST_59 end

	//TEST_54 start
	vector<int> spiralOrder(vector<vector<int>>& matrix) 
	{
		vector<int> ans;
		int row = (int)matrix.size();
		int col = (int)matrix[0].size();
		vector<vector<bool>> visited(row, vector<bool>(col));
		int element = row * col;
		vector<pair<int, int>> direc{{ 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }};
		int index = 0;
		int x = 0;
		int y = 0;
		for (int i = 0; i < element; i++)
		{
			ans.emplace_back(matrix[x][y]);
			visited[x][y] = true;
			int new_x = x + direc[index].first;
			int new_y = y + direc[index].second;
			if ((new_x == row - 1 && new_y == -1) || (new_x == 0 && new_y == col) || (new_x == row && new_y == col - 1) || visited[new_x][new_y])
			{
				index++;
				if (index == 4)
					index = 0;
				new_x = x + direc[index].first;
				new_y = y + direc[index].second;
			}
			x = new_x;
			y = new_y;
		}
		return ans;
	}
	//TEST_54 end

	//TEST_15 start
	//这个是官方的题解，排序加双指针
	vector<vector<int>> threeSum(vector<int>& nums) 
	{
		int n = (int)nums.size();
		sort(nums.begin(), nums.end());
		vector<vector<int>> ans;
		// 枚举 a
		for (int first = 0; first < n; ++first) {
			// 需要和上一次枚举的数不相同
			if (first > 0 && nums[first] == nums[first - 1]) {
				continue;
			}
			// c 对应的指针初始指向数组的最右端
			int third = n - 1;
			int target = -nums[first];
			// 枚举 b
			for (int second = first + 1; second < n; ++second) {
				// 需要和上一次枚举的数不相同
				if (second > first + 1 && nums[second] == nums[second - 1]) {
					continue;
				}
				// 需要保证 b 的指针在 c 的指针的左侧
				while (second < third && nums[second] + nums[third] > target) {
					--third;
				}
				// 如果指针重合，随着 b 后续的增加
				// 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
				if (second == third) {
					break;
				}
				if (nums[second] + nums[third] == target) {
					ans.push_back({ nums[first], nums[second], nums[third] });
				}
			}
		}
		return ans;
	}
	//这个方法时间超时了，shit，但是应该是对的，mb
	void DFS_15(vector<int>& nums, int target, vector<int>& combine, vector<vector<int>>& ans, int idx)
	{
		if (combine.size() == 3)
		{
			if (target == 0)
			{
				ans.emplace_back(combine);
				return;
			}
			else
				return;
		}
		if (idx == nums.size())
			return;

		int once = nums[idx];
		for (int i = idx; i < nums.size(); i++)
		{
			if (i > idx && nums[i] == once)
				continue;
			once = nums[i];
			combine.emplace_back(nums[i]);
			DFS_15(nums, target + nums[i], combine, ans, i + 1);
			combine.pop_back();
		}
	}

	vector<vector<int>> threeSum_time_out(vector<int>& nums) 
	{
		vector<vector<int>> ans;
		vector<int> combine;
		sort(nums.begin(), nums.end());
		DFS_15(nums, 0, combine, ans, 0);
		return ans;
	}
	//TEST_15 end

	//TEST_40 start
	vector<int> candidates;
	vector<vector<int>> res;
	vector<int> path;
	void DFS_40(int start, int target) {
		if (target == 0) {
			res.push_back(path);
			return;
		}

		for (int i = start; i < candidates.size() && target - candidates[i] >= 0; i++) {
			if (i > start && candidates[i] == candidates[i - 1])
				continue;
			path.push_back(candidates[i]);
			DFS_40(i + 1, target - candidates[i]);
			path.pop_back();
		}
	}

	vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
		sort(candidates.begin(), candidates.end());
		this->candidates = candidates;
		DFS_40(0, target);
		return res;
	}
	//TEST_40 end

	//TEST_39 start
	//递归时尽量保证方程参数较少为好，时间空间的消耗都会变少
	void dfs_39(vector<vector<int>>& ans, vector<int>& combine, int target, int sum, vector<int> candidiates, int index)
	{
		if (sum == target)
		{
			ans.push_back(combine);
			return;
		}
		if (index == candidiates.size())
			return;
		if (sum > target)
			return;
		dfs_39(ans, combine, target, sum, candidiates, index + 1);
		combine.push_back(candidiates[index]);
		dfs_39(ans, combine, target, sum + candidiates[index], candidiates, index);
		combine.pop_back();
	}
	vector<vector<int>> combinationSum(vector<int>& candidates, int target) 
	{
		vector<vector<int>> ans;
		vector<int> combine;
		dfs_39(ans, combine, target, 0, candidates, 0);

		return ans;
	}
	//TEST_39 end

	//TEST_2455 start
	int bit_sum(int value)
	{
		int sum = 0;
		while (value / 10)
		{
			sum += value % 10;
			value /= 10;
		}
		return sum + value;
	}
	int averageValue(vector<int>& nums) 
	{
		int sum = 0;
		int num = 0;
		for (int i = 0; i < (int)nums.size(); i++)
		{
			if (nums[i] % 2 == 0 && bit_sum(nums[i]) % 3 == 0)
			{
				sum += nums[i];
				num++;
			}
		}
		if (num == 0)
			return 0;
		return sum / num;
	}
	//TEST_2455 end

	//TEST_313 start
	int nthSuperUglyNumber(int n, vector<int>& primes) 
	{
		vector<long> dp(n + 1);
		int m = (int)primes.size();
		vector<int> pointers(m, 0);
		vector<long> nums(m, 1);
		for (int i = 1; i <= n; i++) {
			long minNum = INT_MAX;
			for (int j = 0; j < m; j++) {
				minNum = min(minNum, nums[j]);
			}
			dp[i] = minNum;
			for (int j = 0; j < m; j++) {
				if (nums[j] == minNum) {
					pointers[j]++;
					nums[j] = dp[pointers[j]] * primes[j];
				}
			}
		}
		return dp[n];
	}
	//TEST_313 end

	//TEST_1201 start
	//这道题tm是数学题，不会做！！！！！！
	/*int nthUglyNumber_new(int n, int a, int b, int c)
	{
		long long ab = lcm((long long)a, (long long)b);
		long long bc = lcm((long long)b, (long long)c);
		long long ac = lcm((long long)a, (long long)c);
		long long abc = lcm((long long)ab, (long long)ac);
		int left = 1, right = (int)2e9;
		auto check = [&](int x) {
			return n <= (long long)x / a + x / b + x / c - x / ab - x / ac - x / bc + x / abc;
		};
		while (left < right) {
			int mid = left + (right - left) / 2;
			if (check(mid)) {
				right = mid;
			}
			else {
				left = mid + 1;
			}
		}
		return left;
	}*/
	//此方法没发过，因为内存超出限制fuckkkkkkkk
	int nthUglyNumber(int n, int a, int b, int c) 
	{
		vector<int> dp(n, 0);
		dp[0] = min(min(a, b), c);
		int index = 1;
		int p1 = 1;
		int p2 = 1;
		int p3 = 1;
		while (n > 1)
		{
			int num1 = p1 * a;
			int num2 = p2 * b;
			int num3 = p3 * c;
			int minVal = min(min(num1, num2), num3);
			if (minVal == dp[index - 1])
			{
				if (minVal == num1)
					p1++;
				else if (minVal == num2)
					p2++;
				else
					p3++;
				continue;
			}
			if (minVal == num1)
				p1++;
			else if (minVal == num2)
				p2++;
			else
				p3++;

			dp[index] = minVal;
			index++;
			n--;
		}

		return dp[index - 1];
	}
	//TEST_1201 end

	//TEST_264 start
	int nthUglyNumber(int n) 
	{
		priority_queue<int, vector<int>, greater<int>> heap;
		unordered_set<int> stock;
		
		heap.push(1);
		stock.insert(1);
		int ans = 0;
		for (int i = 0; i < n; i++)
		{
			ans = heap.top();
			heap.pop();
			int a = ans * 2;
			int b = ans * 3;
			int c = ans * 5;
			if (stock.find(a) == stock.end())
			{
				heap.push(a);
				stock.insert(a);
			}
			if (stock.find(b) == stock.end())
			{
				heap.push(b);
				stock.insert(b);
			}
			if (stock.find(c) == stock.end())
			{
				heap.push(c);
				stock.insert(c);
			}
		}
		return ans;
	}
	//TEST_264 end

	//TEST_95 start
	vector<TreeNode*> real_generateTrees(int start, int end)
	{
		vector<TreeNode*> ans;
		if (start > end)
		{
			ans.push_back(nullptr);
			return ans;
		}
		if (start == end)
		{
			TreeNode* curr = new TreeNode(start);
			ans.push_back(curr);
			return ans;
		}
		for (int i = start; i <= end; i++)
		{
			vector<TreeNode*> left = real_generateTrees(start, i - 1);
			vector<TreeNode*> right = real_generateTrees(i + 1, end);

			for (TreeNode* l: left)
			{
				for (TreeNode* r : right)
				{
					TreeNode* root = new TreeNode(i);
					root->left = l;
					root->right = r;
					ans.push_back(root);
				}
			}
		}
		return ans;
	}

	vector<TreeNode*> generateTrees(int n) 
	{
		return real_generateTrees(1, n);
	}
	//TEST_95 end

	//TEST_96 start
	int numTrees(int n) 
	{
		vector<int> dp(n);
		dp[0] = 1;
		for (int i = 1; i < n; i++)
		{
			dp[i] += 2 * dp[i - 1];
			for (int j = 1; j < i; j++)
			{
				dp[i] += dp[i - j - 1] * dp[j - 1];
			}
		}
		return dp[n - 1];
	}
	//TEST_96 end

	//TEST_14 start
	string longestCommonPrefix(vector<string>& strs) 
	{
		if (strs[0] == "")
			return "";
		int n = (int)strs.size();
		int length = (int)strs[0].length();
		string ans = "";
		for (int i = 0; i < length; i++)
		{
			char curr = strs[0][i];
			for (int j = 1; j < n; j++)
			{
				if (strs[j][i] != curr)
					return ans;
			}
			ans += curr;
		}
		return ans;
	}
	//TEST_14 end

	//TEST_13 start
	int getValue(char a)
	{
		switch (a)
		{
		case 'I':
			return 1;
		case 'V':
			return 5;
		case 'X':
			return 10;
		case 'L':
			return 50;
		case 'C':
			return 100;
		case 'D':
			return 500;
		case 'M':
			return 1000;
		default:
			return -1;
		}
		return -1;
	}

	int romanToInt(string s)
	{
		int maxVal = getValue(s[s.length() - 1]);
		int sum = maxVal;
		for (int i = (int)s.length() - 2; i >= 0; i--)
		{
			int curr = getValue(s[i]);
			if (curr >= maxVal)
			{
				maxVal = curr;
				sum += curr;
			}
			else
			{
				sum -= curr;
			}

		}
		return sum;
	}
	//TEST_13 end

	//TEST_1093 start
	vector<double> sampleStats(vector<int>& count)
	{
		vector<double> ans(5);
		int minVal = INT_MAX;
		int maxVal = INT_MIN;
		long sum = 0;
		bool pass = false;
		int ccount = 0;
		int mode = 0;
		int mode_index = 0;

		for (int i = 0; i < (int)count.size(); i++)
		{
			if (!pass && count[i] != 0)
			{
				minVal = i;
				pass = true;
			}
			if (count[i] != 0)
			{
				if (count[i] > mode)
				{
					mode = count[i];
					mode_index = i;
				}
				sum += count[i] * i;
				maxVal = i;
				ccount += count[i];
			}
		}
		double mean = (double)sum / ccount;
		double median = 0;

		int index = ccount / 2;
		int count_index = 0;
		while (index > 0)
		{
			index -= count[count_index];
			count_index++;
		}
		if (index == 0)
		{
			if (ccount % 2 == 0)
				median += count_index - 1;
			for (count_index; count_index < (int)count.size(); count_index++)
			{
				if (count[count_index] != 0)
					break;
			}
			if (ccount % 2 == 0)
				median = (median + count_index) / 2;
			else
				median = count_index;
		}
		else
			median = count_index - 1;
		ans[0] = minVal;
		ans[1] = maxVal;
		ans[2] = mean;
		ans[3] = median;
		ans[4] = mode_index;
		return ans;
	}
	//TEST_1093 end

	//TEST_173 start
	int countMatches(vector<vector<string>>& items, string ruleKey, string ruleValue)
	{
		int index;
		if (ruleKey == "type")
			index = 0;
		else if (ruleKey == "color")
			index = 1;
		else
			index = 2;

		int sum = 0;
		for (int i = 0; i < (int)items.size(); i++)
		{
			if (items[i][index] == ruleValue)
				sum++;
		}
		return sum;
	}
	//TEST_173 end

	//TEST_1689 start
	int minPartitions(string n)
	{
		int mx = (int)n[0] - 48;
		for (int i = 0; i < (int)n.length(); i++)
		{
			if ((int)n[i] - 48 > mx)
				mx = (int)n[i] - 48;
		}
		return mx;
	}
	//TEST_1689 end

	//TEST_276 start
	const long long k1 = 1117;
	const long long k2 = (long long)1e9 + 7;
	unordered_map<long long, string> dataBase;
	unordered_map<string, long long> urlToKey;
	string encode(string longUrl) {
		if (urlToKey.count(longUrl) > 0) {
			return string("http://tinyurl.com/") + to_string(urlToKey[longUrl]);
		}
		long long key = 0, base = 1;
		for (auto c : longUrl) {
			key = (key + c * base) % k2;
			base = (base * k1) % k2;
		}
		while (dataBase.count(key) > 0) {
			key = (key + 1) % k2;
		}
		dataBase[key] = longUrl;
		urlToKey[longUrl] = key;
		return string("http://tinyurl.com/") + to_string(key);
	}

	string decode(string shortUrl) {
		int p = (int)shortUrl.rfind('/') + 1;
		int key = stoi(shortUrl.substr(p, int(shortUrl.size()) - p));
		return dataBase[key];
	}
	//TEST_276 end

	//TEST_807 start
	//暴力法，傻逼都能想出来，不写了
	int maxIncreaseKeepingSkyline(vector<vector<int>>& grid)
	{
		return 0;
	}
	//TEST_807 end

	//TEST_1828 start
	//第二种方法是根据x轴排序然后用二分查找缩小范围再计算，不改变时间复杂度
	vector<int> countPoints(vector<vector<int>>& points, vector<vector<int>>& queries)
	{
		int n = (int)queries.size();
		vector<int> ans(n);

		for (int i = 0; i < n; i++)
		{
			int curr = 0;
			for (int j = 0; j < (int)points.size(); j++)
			{
				double dist = sqrt(pow(abs(points[j][0] - queries[i][0]), 2) + pow(abs(points[j][1] - queries[i][1]), 2));
				if (dist <= queries[i][2])
				{
					curr++;
				}
			}

			ans[i] = curr;
		}
		return ans;
	}
	//TEST_1828 end

	//TEST_2011 start
	int finalValueAfterOperations(vector<string>& operations)
	{
		int sum = 0;
		for (int i = 0; i < (int)operations.size(); i++)
		{
			if (operations[i][0] == '+')
				sum++;
			else if (operations[i][0] == '-')
				sum--;
			else if (operations[i][2] == '+')
				sum++;
			else
				sum--;
		}
		return sum;
	}
	//TEST_2011 end

	//TEST_1769 start
	vector<int> minOperations(string boxes)
	{
		int n = (int)boxes.length();
		vector<int> ans(n, 0);
		int right = 0;
		int left = 0;
		for (int i = 0; i < n; i++)
		{
			if (boxes[i] == '1')
			{
				ans[0] += i;
				right++;
			}

		}
		if (boxes[0] == '1')
		{
			left++;
			right--;
		}
		for (int i = 1; i < n; i++)
		{
			ans[i] = ans[i - 1] - right + left;
			if (boxes[i] == '1')
			{
				left++;
				right--;
			}
		}
		for (int i = 0; i < n; i++)
			cout << ans[i] << endl;
		return ans;
	}
	//TEST_1769 end

	//TEST_2446 start
	int convert(string a)
	{
		return a[0] * 600 + a[1] * 60 + a[3] * 10 + a[4];
	}

	bool haveConflict(vector<string>& event1, vector<string>& event2)
	{
		string event_11 = event1[0];
		string event_12 = event1[1];
		string event_21 = event2[0];
		string event_22 = event2[1];

		int e_11 = convert(event_11);
		int e_12 = convert(event_12);
		int e_21 = convert(event_21);
		int e_22 = convert(event_22);

		if (e_11 == e_21 || e_12 == e_22)
			return 1;
		if (e_11 < e_21 && e_12 >= e_21)
			return 1;
		if (e_21 < e_11 && e_22 >= e_11)
			return 1;

		return 0;
	}
	//TEST_2446 end

	//TEST_1335 start
	int minDifficulty(vector<int>& jobDifficulty, int d)
	{
		size_t n = jobDifficulty.size();
		if (d > n)
			return -1;

		vector<vector<int>> dp(n + 1, vector<int>(d + 1, 99999));

		for (int i = 1, mx = 0; i < n + 1; i++)
		{
			mx = max(mx, jobDifficulty[i - 1]);
			dp[i][1] = mx;
		}

		for (int i = 1; i < n + 1; i++)
		{
			for (int j = 2; j < d + 1; j++)
			{
				for (int k = i, mx = 0; k > 0; k--)
				{
					mx = max(mx, jobDifficulty[k - 1]);
					dp[i][j] = min(dp[i][j], dp[k - 1][j - 1] + mx);
				}
			}
		}

		if (dp[n][d] == 99999)
			return -1;
		return dp[n][d];

	}
	//TEST_1335 end

	//TEST_1072 start
	int maxEqualRowsAfterFlips(vector<vector<int>>& matrix)
	{
		multiset<string> temp;
		for (int i = 0; i < (int)matrix.size(); ++i)
		{
			string curr = "";
			for (int j = 0; j < (int)matrix[0].size(); ++j)
			{
				if (matrix[i][0] == 0)
				{
					curr += char(matrix[i][j] ^ 1);
				}
				else
				{
					curr += char(matrix[i][j]);
				}
			}

			temp.insert(curr);
		}
		int maxL = 0;
		auto first = temp.begin();
		auto second = temp.begin();
		second++;
		int curL = 1;
		while (second != temp.end())
		{

			if (*first == *second)
			{

				second++;
				curL++;
			}
			else
			{
				curL = 1;
				first = second;
				second++;
			}
			if (curL > maxL)
				maxL = curL;
		}

		return maxL;
	}
	//TEST_1072 end

	//TEST_292 start
	bool canWinNim(int n)
	{
		return (n % 4) != 0;
	}
	//TEST_292 end

	//TEST_119 start
	vector<int> getRow(int rowIndex) {
		vector<int> pre, cur;
		for (int i = 0; i <= rowIndex; ++i) {
			cur.resize(i + 1);
			cur[0] = cur[i] = 1;
			for (int j = 1; j < i; ++j) {
				cur[j] = pre[j - 1] + pre[j];
			}
			pre = cur;
		}
		return pre;
	}
	//TEST_199 end

	//TEST_118 start
	vector<vector<int>> generate(int numRows)
	{
		vector<vector<int>> dp(numRows);
		vector<int> a{ 1 };
		vector<int> b{ 1, 1 };
		if (numRows == 1)
		{
			dp[0] = a;
			return dp;
		}
		if (numRows == 2)
		{
			dp[0] = a;
			dp[1] = b;
			return dp;
		}

		dp[0] = a;
		dp[1] = b;
		for (int i = 2; i < numRows; i++)
		{
			for (int j = 0; j <= i; j++)
			{
				if (j == 0 || j == i)
					dp[i].push_back(1);
				else
				{
					dp[i].push_back(dp[i - 1][j - 1] + dp[i - 1][j]);
				}
			}
		}
		return dp;
	}
	//TEST_118 end

	//TEST_63 start
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid)
	{
		int m = (int)obstacleGrid.size();
		int n = (int)obstacleGrid[0].size();
		bool curr = false;
		vector<vector<int>> dp(m, vector<int>(n));

		int i;
		for (i = 0; i < m; i++)
		{
			if (curr || obstacleGrid[i][0])
			{
				dp[i][0] = 0;
				curr = true;
			}
			else
				dp[i][0] = 1;
		}
		curr = false;
		for (i = 0; i < n; i++)
		{
			if (curr || obstacleGrid[0][i])
			{
				dp[0][i] = 0;
				curr = true;
			}
			else
				dp[0][i] = 1;
		}


		for (i = 1; i < m; i++)
		{
			for (int j = 1; j < n; j++)
			{
				if (obstacleGrid[i][j])
					dp[i][j] = 0;
				else
					dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
		return dp[m - 1][n - 1];
	}
	//TEST_63 end

	//TEST_62 start
	int uniquePaths(int m, int n)
	{
		vector<vector<int>> dp(m, vector<int>(n));
		int i;
		for (i = 0; i < m; i++)
			dp[i][0] = 1;
		for (i = 0; i < n; i++)
			dp[0][i] = 1;

		for (i = 1; i < m; i++)
		{
			for (int j = 1; j < n; j++)
			{
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}

		return dp[m - 1][n - 1];
	}

	//TEST_62 end

	//TEST_1091 start
	int shortestPathBinaryMatrix(vector<vector<int>>& grid)
	{
		int n = (int)grid.size();
		if (grid[0][0] == 1)
			return -1;
		vector<vector<int>> distance(n, vector<int>(n, -1));
		queue<pair<int, int>> map;
		map.push(make_pair(0, 0));
		distance[0][0] = 1;

		while (!map.empty())
		{
			pair<int, int> curr = map.front();
			map.pop();

			if (curr.first == n - 1 && curr.second == n - 1)
				return distance[n - 1][n - 1];

			for (int dx = -1; dx < 2; dx++)
			{
				for (int dy = -1; dy < 2; dy++)
				{
					if (dx == 0 && dy == 0)
						continue;
					int new_x = curr.first + dx;
					int new_y = curr.second + dy;
					if (new_x >= n || new_x < 0 || new_y >= n || new_y < 0)
						continue;
					if (grid[new_x][new_y] == 1 || distance[new_x][new_y] != -1)
						continue;

					distance[new_x][new_y] = distance[curr.first][curr.second] + 1;
					map.push(make_pair(new_x, new_y));
				}
			}
		}
		return -1;
	}
	//TEST_1091 end

	//TEST_41 start
	int firstMissingPositive(vector<int>& nums) {
		for (int i = 0; i < nums.size(); ++i)
		{
			while (nums[i] > 0 && nums[i] <= nums.size() && nums[i] == (i + 1))
			{
				int temp = nums[nums[i] - 1];
				nums[nums[i] - 1] = nums[i];
				nums[i] = temp;
			}
		}
		int i;
		for (i = 0; i < nums.size(); ++i)
		{
			if (nums[i] != (i + 1))
				return (i + 1);
		}
		return i + 1;
	}
	//TEST_41 end

	//TEST_22 start
	void dfs(vector<string>& ans, int left, int right, string curr)
	{
		if (left < 0 || right < 0)
			return;
		if (left == 0 && right == 0)
			ans.push_back(curr);

		if (left > right)
			return;

		dfs(ans, left - 1, right, curr + "(");
		dfs(ans, left, right - 1, curr + ")");
	}

	vector<string> generateParenthesis(int n) {
		vector<string> ans;

		dfs(ans, n, n, "");

		return ans;
	}
	//TEST_22 end

	//TEST_714 start
	int maxProfit(vector<int>& prices, int fee)
	{
		int n = (int)prices.size();
		vector<vector<int>> dp(n, vector<int>(2));
		dp[0][0] = 0;
		dp[0][1] = -prices[0];

		for (int i = 1; i < n; i++)
		{
			dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
			dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
		}

		return dp[n - 1][0];
	}
	//TEST_714 end

	//TEST_309 start
	int maxProfit_4(vector<int>& prices)
	{
		int n = (int)prices.size();
		vector<vector<int>>dp(n, vector<int>(3));
		dp[0][0] = -prices[0];
		dp[0][1] = 0;
		dp[0][2] = 0;

		for (int i = 1; i < n; i++)
		{
			dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i]);
			dp[i][1] = dp[i - 1][0] + prices[i];
			dp[i][2] = max(dp[i - 1][1], dp[i - 1][2]);
			cout << dp[i][0] << " " << dp[i][1] << " " << dp[i][2] << endl;
		}

		return max(dp[n - 1][1], dp[n - 1][2]);
	}
	//TEST_309 end

	//TEST_2451 start
	string oddString(vector<string>& words)
	{
		int length = (int)words[0].length();
		int n = (int)words.size();
		map<vector<int>, int> stock;
		map<vector<int>, int> index;

		for (int i = 0; i < n; i++)
		{
			vector<int> temp;
			for (int j = 1; j < length; j++)
			{
				temp.push_back(words[i][j] - words[i][j - 1]);
			}
			index[temp] = i;
			stock[temp]++;
		}

		for (auto i = stock.begin(); i != stock.end(); i++)
		{
			if (i->second == 1)
			{
				return words[index[i->first]];
			}
		}
		return "";
	}
	//TEST_2451 end

	//TEST_338 start
	vector<int> countBits(int n)
	{
		vector<int> dp(n + 1);
		dp[0] = 0;
		if (n == 0)
			return dp;
		dp[1] = 1;
		if (n == 1)
			return dp;
		dp[2] = 1;
		if (n == 2)
			return dp;

		for (int i = 3; i < n + 1; i++)
		{
			if (i % 2 == 1)
				dp[i] = dp[i / 2] + 1;
			else
				dp[i] = dp[i / 2];
		}

		return dp;
	}
	//TEST_338 end

	//TEST_123 start
	int maxProfit_3(vector<int>& prices)
	{
		int n = (int)prices.size();
		vector<vector<int>> dp(n, vector<int>(4));

		dp[0][0] = -prices[0];
		dp[0][1] = 0;
		dp[0][2] = -prices[0];
		dp[0][3] = 0;

		for (int i = 1; i < n; i++)
		{
			dp[i][0] = max(dp[i - 1][0], -prices[i]);
			dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
			dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] - prices[i]);
			dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] + prices[i]);
		}

		return dp[n - 1][3];
	}
	//TEST_123 end

	//TEST_122 start
	//分为持有股票和不持有股票两种状态
	int maxProfit_2(vector<int>& prices)
	{
		int n = (int)prices.size();
		vector<vector<int>> dp(n, vector<int>(2));
		dp[0][0] = 0;
		dp[0][1] = -prices[0];

		for (int i = 1; i < n; i++)
		{
			dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
			dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
		}

		return dp[n - 1][0];
	}
	//TEST_122 end

	//TEST_121 start
	int maxProfit_1(vector<int>& prices)
	{
		int n = (int)prices.size();
		vector<int> dp(n);
		dp[0] = 0;
		int min = prices[0];
		for (int i = 1; i < n; i++)
		{
			if (prices[i] < min)
				min = prices[i];
			dp[i] = max(dp[i - 1], prices[i] - min);
		}

		return dp[n - 1];
	}
	//TEST_121 end

	//TEST_5 start
	int maxLength(string s, int begin, int end)
	{
		int diff = 0;
		while (begin >= 0 && end < s.length() && s[begin] == s[end])
		{
			--begin;
			++end;
			++diff;
		}

		return diff;
	}

	string longestPalindrome_1(string s)
	{
		int maxL = 0;
		int begin = 0;
		for (int i = 0; i < s.length(); ++i)
		{
			int odd = maxLength(s, i, i);
			int even = maxLength(s, i, i + 1);
			if ((odd * 2 - 1) > maxL)
			{
				maxL = odd * 2 - 1;
				begin = i - odd + 1;
			}
			if ((even * 2) > maxL)
			{
				maxL = even * 2;
				begin = i - even + 1;
			}
		}

		return s.substr(begin, maxL);
	}
	//TEST_5 end

	//TEST_3 start
	int lengthOfLongestSubstring(string s) {
		if (s.size() == 0) return 0;
		unordered_set<char> lookup;
		int maxStr = 0;
		int left = 0;
		for (int i = 0; i < s.size(); i++) {
			while (lookup.find(s[i]) != lookup.end()) {
				lookup.erase(s[left]);
				left++;
			}
			maxStr = max(maxStr, i - left + 1);
			lookup.insert(s[i]);
		}
		return maxStr;
	}
	//TEST_3 end

	//TEST_2 start
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
		bool plus_one = false;
		ListNode* ans = new ListNode();
		ListNode* ptr_ans = ans;
		while (l1 || l2)
		{
			int l1_val, l2_val, sum;
			if (l1)
				l1_val = l1->val;
			else
				l1_val = 0;
			if (l2)
				l2_val = l2->val;
			else
				l2_val = 0;

			if (plus_one)
				sum = l1_val + l2_val + 1;
			else
				sum = l1_val + l2_val;

			if (sum >= 10)
			{
				sum -= 10;
				plus_one = true;
			}
			else
				plus_one = false;

			ListNode* temp = new ListNode(sum);
			ans->next = temp;
			ans = ans->next;

			if (l1)
				l1 = l1->next;
			if (l2)
				l2 = l2->next;

		}
		if (plus_one)
		{
			ListNode* temp = new ListNode(1);
			ans->next = temp;
		}

		return ptr_ans->next;
	}
	//TEST_2 end

	//TEST_1090 start 
	//排列下标是更优解
	int largestValsFromLabels(vector<int>& values, vector<int>& labels, int numWanted, int useLimit)
	{
		auto c = [](pair<int, int> a, pair<int, int> b)->bool {return a.first > b.first; };
		vector<pair<int, int>> temp;
		map<int, int> stock;
		map<int, int> limit;
		int n = (int)values.size();
		int sum = 0;


		for (int i = 0; i < n; i++)
		{
			temp.push_back(make_pair(values[i], labels[i]));
			stock[labels[i]]++;
			limit[labels[i]] = useLimit;
		}
		sort(temp.begin(), temp.end(), c);

		int index = 0;
		while (index != n && numWanted != 0)
		{
			if (stock[temp[index].second] > 0 && limit[temp[index].second] > 0)
			{
				sum += temp[index].first;
				stock[temp[index].second]--;
				limit[temp[index].second]--;
				numWanted--;
			}
			index++;
		}

		return sum;
	}
	//TEST_1090 end

	//TEST_409 start
	int longestPalindrome_2(string s)
	{
		unordered_map<char, int> stock;
		int n = (int)s.length();
		int sum = 0;
		if (n == 1)
			return 1;
		for (int i = 0; i < s.length(); i++)
			stock[s[i]]++;
		for (int i = 97; i < 97 + 26; i++)
		{
			if (stock.find(i) != stock.end())
			{
				sum += stock[i] / 2 * 2;
			}
		}
		for (int i = 65; i < 65 + 26; i++)
		{
			if (stock.find(i) != stock.end())
			{
				sum += stock[i] / 2 * 2;
			}
		}
		if (sum != s.length())
			sum++;
		return sum;
	}
	//TEST_409 end

	//TEST_605 start
	bool canPlaceFlowers(vector<int>& flowerbed, int n)
	{
		if (n == 0)
			return true;
		flowerbed.insert(flowerbed.begin(), 0);
		flowerbed.insert(flowerbed.end(), 0);
		int number = (int)flowerbed.size();
		for (int i = 0; i < number - 2; i++)
		{
			if (flowerbed[i] == 0 && flowerbed[i + 1] == 0 && flowerbed[i + 2] == 0)
			{
				n--;
				i++;
			}
			if (n == 0)
				return true;
		}
		return false;
	}
	//TEST_605 end

	//TEST_11 start\
	    //贪心的核心就是数学推导，通过推导减少不必要的计算过程，优化时间复杂度
	int maxArea(vector<int>& height)
	{
		int left = 0;
		int right = (int)height.size() - 1;
		int max = INT_MIN;
		while (left != right)
		{
			int curr = min(height[left], height[right]) * (right - left);
			if (curr > max)
				max = curr;
			if (height[left] > height[right])
				right--;
			else
				left++;
		}
		return max;
	}
	//TEST_11 end

	//TEST_98 start
	bool s()
	{
		return 0;
	}

	bool isValidBST(TreeNode* root)
	{

	}
	//TEST_98 end

	//TEST_LCP_33 start
	int storeWater(vector<int>& bucket, vector<int>& vat)
	{
		int n = int(bucket.size());

		int ans = INT_MAX;
		for (int i = 1; i < 100000; i++)
		{
			int sum = 0;
			for (int j = 0; j < n; j++)
			{
				sum += max(0, (int)(ceil(vat[j] / (double)i) - bucket[j]));
			}
			ans = min(ans, sum + i);
		}

		return ans;
	}
	//TEST_LCP_33 end

	//TEST_1373 start  
	int maxSumBST(TreeNode* root)
	{
		return 0;
	}
	//TEST_1373 end

	//TEST_1079 start
	int numTilePossibilities(string tiles) {
		unordered_map<char, int> count;
		set<char> tile;
		int n = int(tiles.length());
		for (char c : tiles) {
			count[c]++;
			tile.insert(c);
		}
		return dfs(count, tile, n) - 1;
	}

	int dfs(unordered_map<char, int>& count, set<char>& tile, int i) {
		if (i == 0) {
			return 1;
		}
		int res = 1;
		for (char t : tile) {
			if (count[t] > 0) {
				count[t]--;
				res += dfs(count, tile, i - 1);
				count[t]++;
			}
		}
		return res;
	}
	//TEST_1079 end

	//TEST_109 start
	//摩尔计数法！！！
	int majorityElement(vector<int>& nums)
	{
		int candidate = -1;
		int count = 0;
		for (int num : nums)
		{
			if (num == candidate)
				++count;
			else if (--count < 0)
			{
				candidate = num;
				count = 1;
			}
		}
		return candidate;
	}
	//TEST_169 end

	//TEST_88 start
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
		int i = int(nums1.size()) - 1;
		m--;
		n--;
		while (n >= 0) {
			while (m >= 0 && nums1[m] > nums2[n]) {
				swap(nums1[i--], nums1[m--]);
			}
			swap(nums1[i--], nums2[n--]);
		}
	}
	//TEST_88 end

	//TEST_1073 start
	vector<int> addNegabinary(vector<int>& arr1, vector<int>& arr2) {
		int i = int(arr1.size() - 1);
		int j = int(arr2.size() - 1);
		int carry = 0;
		vector<int> ans;
		while (i >= 0 || j >= 0 || carry) {
			int x = carry;
			if (i >= 0) {
				x += arr1[i];
			}
			if (j >= 0) {
				x += arr2[j];
			}

			if (x >= 2) {
				ans.push_back(x - 2);
				carry = -1;
			}
			else if (x >= 0) {
				ans.push_back(x);
				carry = 0;
			}
			else {
				ans.push_back(1);
				carry = 1;
			}
			--i;
			--j;
		}
		while (ans.size() > 1 && ans.back() == 0) {
			ans.pop_back();
		}
		reverse(ans.begin(), ans.end());
		return ans;
	}
	//TEST_1073 end

	//TEST_1017 start
	string baseNeg2(int n) {
		string ans;
		while (n)
		{
			int remain = n % (-2);
			ans += '0' + abs(remain);
			n = remain < 0 ? n / (-2) + 1 : n / (-2);
		}
		reverse(ans.begin(), ans.end());
		return ans.empty() ? "0" : ans;
	}
	//TEST_1017 end
};
#endif