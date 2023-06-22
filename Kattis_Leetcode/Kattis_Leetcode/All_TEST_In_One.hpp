#ifndef ALL_TEST
#define ALL_TEST

#include "All_Header.hpp"

using namespace std;
class Solution {
public:
	//TEST_16.19 start
	//为了避免重复添加元素在队列，需要在添加队列的时候就标记visited，此处即为修改元素值为-1
	vector<int> pondSizes(vector<vector<int>>& land) 
	{
		int row = (int)land.size();
		int col = (int)land[0].size();
		vector<vector<int>> dir{{-1, 0}, { 1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }};
		vector<int> ans;
		
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				if (land[i][j] == 0)
				{
					queue<pair<int, int>> stock;
					int curr = 0;
					stock.push(make_pair(i, j));
					land[i][j] = -1;
					while (!stock.empty())
					{
						pair<int, int> a = stock.front();
						stock.pop();
						curr++;	
						int dx, dy;
						for (int i = 0; i < 8; i++)
						{
							dx = a.first + dir[i][0];
							dy = a.second + dir[i][1];
							if (dx >= 0 && dx < row && dy >= 0 && dy < col && land[dx][dy] == 0)
							{
								land[dx][dy] = -1;
								stock.push(make_pair(dx, dy));
							}
						}
					}
					ans.push_back(curr);
				}
			}
		}
		sort(ans.begin(), ans.end());
		return ans;
	}
	//TEST_16.19 end

	//TEST_1143 start
	int longestCommonSubsequence(string text1, string text2) 
	{
		int n1 = text1.length();
		int n2 = text2.length();
		vector<vector<int>>dp(n1 + 1, vector<int>(n2 + 1, 0));
		for (int i = 1; i <= n1; i++)
		{
			char c1 = text1[i - 1];
			for (int j = 1; j <= n2; j++)
			{
				char c2 = text2[j - 1];
				if (c1 == c2)
				{
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}
				else
				{
					dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		return dp[n1][n2];
	}
	//TEST_1143 end

	//TEST_53 start
	//优化空间
	int maxSubArray_SPACE(vector<int>& nums) 
	{
		int pre = 0, maxAns = nums[0];
		for (const auto& x : nums) 
		{
			pre = max(pre + x, x);
			maxAns = max(maxAns, pre);
		}
		return maxAns;
	}
	int maxSubArray(vector<int>& nums)
	{
		int n = (int)nums.size();
		int ans = INT_MIN;
		vector<int>dp(n);
		dp[0] = nums[0];
		for (int i = 1; i < n; i++)
		{
			dp[i] = max(dp[i - 1] + nums[i], nums[i]);
		}
		for (int i = 0; i < n; i++)
		{
			ans = max(ans, dp[i]);
		}
		return ans;
	}
	//TEST_53 end

	//TEST_LCP_41 start
	bool judge_41(vector<string>& chessboard, int x, int y, int dx, int dy)
	{
		while (x >= 0 && x < (int)chessboard.size() && y >= 0 && y < (int)chessboard[0].size())
		{
			if (chessboard[x][y] == '.')
				return false;
			if (chessboard[x][y] == 'X')
				return true;
			x += dx;
			y += dy;
		}
		return false;
	}

	int BFS_41(vector<string> chessboard, int x, int y, vector<vector<int>>& dir)
	{
		int ans = 0;
		queue<pair<int, int>> stock;
		stock.emplace(make_pair(x, y));
		while (!stock.empty())
		{
			pair<int, int> curr = stock.front();
			stock.pop();
			int dx, dy;
			int currr = 0;
			for (int i = 0; i < 8; i++)
			{
				dx = curr.first + dir[i][0];
				dy = curr.second + dir[i][1];
				if (judge_41(chessboard, dx, dy, dir[i][0], dir[i][1]))
				{
					while (chessboard[dx][dy] != 'X')
					{
						stock.emplace(make_pair(dx, dy));
						chessboard[dx][dy] = 'X';
						currr++;
						dx += dir[i][0];
						dy += dir[i][1];
					}
				}
			}
			ans += currr;
		}
		return ans;
	}

	int flipChess(vector<string>& chessboard) 
	{
		int ans = INT_MIN;
		int row = (int)chessboard.size();
		int col = (int)chessboard[0].size();
		vector<vector<int>> dir{{-1, 0}, { 1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }};

		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				if (chessboard[i][j] == '.')
				{
					ans = max(ans, BFS_41(chessboard, i, j, dir));
				}
			}
		}
		return ans;
	}
	//TEST_LCP_41 end

	//TEST_1595 start
	//递推
	int connectTwoGroups_DP(vector<vector<int>>& cost) 
	{
		int n = (int)cost.size(), m = (int)cost[0].size();
		vector<int> min_cost(m, INT_MAX);
		for (int j = 0; j < m; j++)
			for (auto& c : cost)
				min_cost[j] = min(min_cost[j], c[j]);

		vector<vector<int>> f(n + 1, vector<int>(1ULL << m));
		for (int j = 0; j < 1 << m; j++)
			for (int k = 0; k < m; k++)
				if (j >> k & 1) // 第二组的点 k 未连接
					f[0][j] += min_cost[k]; // 去第一组找个成本最小的点连接

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < 1 << m; j++) {
				int res = INT_MAX;
				for (int k = 0; k < m; k++) // 第一组的点 i 与第二组的点 k
					res = min(res, f[i][j & ~(1 << k)] + cost[i][k]);
				f[i + 1][j] = res;
			}
		}
		return f[n][(1 << m) - 1];
	}
	//回溯
	int connectTwoGroups(vector<vector<int>>& cost) 
	{
		int n = (int)cost.size(), m = (int)cost[0].size();
		vector<int> min_cost(m, INT_MAX);
		for (int j = 0; j < m; j++)
			for (auto& c : cost)
				min_cost[j] = min(min_cost[j], c[j]);

		vector<vector<int>> memo(n, vector<int>(1ULL << m, INT_MAX));
		function<int(int, int)> dfs = [&](int i, int j) -> int {
			if (i < 0) {
				int res = 0;
				for (int k = 0; k < m; k++)
					if (j >> k & 1) // 第二组的点 k 未连接
						res += min_cost[k]; // 去第一组找个成本最小的点连接
				return res;
			}
			int& res = memo[i][j]; // 注意这里是引用
			if (res != INT_MAX) return res; // 之前算过了
			for (int k = 0; k < m; k++) // 第一组的点 i 与第二组的点 k
				res = min(res, dfs(i - 1, j & ~(1 << k)) + cost[i][k]);
			return res;
		};
		return dfs(n - 1, (1 << m) - 1);
	}
	//TEST_1595 end

	//TEST_662 start
	void DFS_662(TreeNode* node, unsigned long long nodeIndex, unsigned long long& ans, int level, unordered_map<unsigned long long, unsigned long long>& stock)
	{
		if (node == nullptr)
			return;
		if (!stock.count(level))
			stock[level] = nodeIndex;
		ans = max(ans, nodeIndex - stock[level] + 1);
		DFS_662(node->left, nodeIndex * 2, ans, level + 1, stock);
		DFS_662(node->right, nodeIndex * 2 + 1, ans, level + 1, stock);
	}

	int widthOfBinaryTree(TreeNode* root) 
	{
		unsigned long long ans = 0;
		unordered_map<unsigned long long, unsigned long long> stock;
		DFS_662(root, 1, ans, 0, stock);
		return (int)ans;
	}
	//TEST_662 end

	//TEST_518 start
	int change(int amount, vector<int>& coins) {
		vector<int> dp(amount + 1, 0);
		dp[0] = 1;
		for (int i = 0; i < (int)coins.size(); i++) 
		{ // 遍历物品
			for (int j = coins[i]; j <= amount; j++) 
			{ // 遍历背包
				dp[j] += dp[j - coins[i]];
				cout << dp[j] << " ";
			}
			cout << endl;
		}
		return dp[amount];
	}
	//TEST_518 end

	//TEST_463 start
	int islandPerimeter(vector<vector<int>>& grid) 
	{
		int row = (int)grid.size();
		int col = (int)grid[0].size();
		int ans = 0;
		bool visited = 0;
		queue<pair<int, int>> q;
		vector<vector<bool>> pass(row, vector<bool>(col, 0));
		for (int i = 0; i < row && !visited; i++)
		{
			for (int j = 0; j < col && !visited; j++)
			{
				if (grid[i][j])
				{
					visited = 1;
					q.push(make_pair(i, j));
					while (!q.empty())
					{
						int len = 4;
						pair<int, int> curr = q.front();
						q.pop();
						if (pass[curr.first][curr.second])
							continue;
						pass[curr.first][curr.second] = 1;
						int dx = curr.first + 1;
						int dy = curr.second;
						if (dx < row && grid[dx][dy])
						{
							if(!pass[dx][dy])
								q.push(make_pair(dx, dy));
							len--;
						}
						dx = curr.first - 1;
						dy = curr.second;
						if (dx >= 0 && grid[dx][dy])
						{
							if (!pass[dx][dy])
								q.push(make_pair(dx, dy));
							len--;
						}
						dx = curr.first;
						dy = curr.second + 1;
						if (dy < col && grid[dx][dy])
						{
							if (!pass[dx][dy])
								q.push(make_pair(dx, dy));
							len--;
						}
						dx = curr.first;
						dy = curr.second - 1;
						if (dy >= 0 && grid[dx][dy])
						{
							if (!pass[dx][dy])
								q.push(make_pair(dx, dy));
							len--;
						}
						cout << len << endl;
						ans += len;
					}
				}
			}
		}
		return ans;
	}
	//TEST_463 end

	//TEST_200 start
	void DFS_200(vector<vector<char>>& grid, int r, int c) 
	{
		int nr = (int)grid.size();
		int nc = (int)grid[0].size();

		grid[r][c] = '0';
		if (r - 1 >= 0 && grid[r - 1][c] == '1') DFS_200(grid, r - 1, c);
		if (r + 1 < nr && grid[r + 1][c] == '1') DFS_200(grid, r + 1, c);
		if (c - 1 >= 0 && grid[r][c - 1] == '1') DFS_200(grid, r, c - 1);
		if (c + 1 < nc && grid[r][c + 1] == '1') DFS_200(grid, r, c + 1);
	}
	int numIslands(vector<vector<char>>& grid) {
		int nr = (int)grid.size();
		if (!nr) return 0;
		int nc = (int)grid[0].size();

		int num_islands = 0;
		for (int r = 0; r < nr; ++r) 
		{
			for (int c = 0; c < nc; ++c) 
			{
				if (grid[r][c] == '1') 
				{
					++num_islands;
					DFS_200(grid, r, c);
				}
			}
		}

		return num_islands;
	}
	//TEST_200 end

	//TEST_1254 start
	int closedIsland(vector<vector<int>>& grid) 
	{
		int row = (int)grid.size();
		int col = (int)grid[0].size();
		int ans = 0;
		vector<vector<bool>> visited(row, vector<bool>(col, 0));
		for (int i = 1; i < row - 1; i++)
		{
			for (int j = 1; j < col - 1; j++)
			{
				if (!grid[i][j] && !visited[i][j])
				{
					queue<pair<int, int>> q;
					q.push(make_pair(i, j));
					bool close = 1;
					while (!q.empty())
					{
						pair<int, int> a = q.front();
						visited[a.first][a.second] = 1;
						if (!grid[i][j] && (a.first == 0 || a.first == row - 1 || a.second == 0 || a.second == col - 1))
							close = 0;
						q.pop();
						int dx = a.first + 1;
						int dy = a.second;
						if (dx >= 0 && dy >= 0 && dx < row && dy < col && !grid[dx][dy] && !visited[dx][dy])
						{
							q.push(make_pair(dx, dy));
						}
						dx = a.first - 1;
						dy = a.second;
						if (dx >= 0 && dy >= 0 && dx < row && dy < col && !grid[dx][dy] && !visited[dx][dy])
						{
							q.push(make_pair(dx, dy));
						}
						dx = a.first;
						dy = a.second + 1;
						if (dx >= 0 && dy >= 0 && dx < row && dy < col && !grid[dx][dy] && !visited[dx][dy])
						{
							q.push(make_pair(dx, dy));
						}
						dx = a.first;
						dy = a.second - 1;
						if (dx >= 0 && dy >= 0 && dx < row && dy < col && !grid[dx][dy] && !visited[dx][dy])
						{
							q.push(make_pair(dx, dy));
						}
					}
					if (close)
						ans++;
				}
			}
		}
		return ans;
	}
	//TEST_1254 end

	//TEST_416 start
	bool canPartition(vector<int>& nums) 
	{
		int target = 0;
		int n = (int)nums.size();
		for (int i : nums)
			target += i;
		if (target & 1)
			return 0;
		target /= 2;
		
		vector<bool> dp(target + 1, 0);
		if (nums[0] <= target)
			dp[nums[0]] = 1;
		for (int i = 1; i < n; i++)
		{
			for (int j = target; j >= 0; j--)
			{
				if (nums[i] == j)
					dp[j] = 1;
				else if (nums[i] < j)
					dp[j] = dp[j] || dp[j - nums[i]];
			}
		}
		return dp[target];
	}
	//TEST_416 end

	//TEST_322 start
	//DP
	int coinChange_DP(vector<int>& coins, int amount)
	{
		int Max = amount + 1;
		vector<int> dp(amount + 1, Max);
		dp[0] = 0;
		for (int i = 1; i <= amount; ++i) 
		{
			for (int j = 0; j < (int)coins.size(); ++j) 
			{
				if (coins[j] <= i) 
				{
					dp[i] = min(dp[i], dp[i - coins[j]] + 1);
				}
			}
		}
		return dp[amount] > amount ? -1 : dp[amount];
	}
	//记忆化搜索？可能吧，但是超时了，麻痹
	int DFS_322(vector<int> coins, int amount, int idx, vector<vector<int>>& cache)
	{
		if (idx < 0)
		{
			if (amount == 0)
				return 0;
			else
				return 9999;
		}
		if (cache[idx][amount] != -1)
			return cache[idx][amount];
		if (amount < coins[idx])
		{
			cache[idx][amount] = DFS_322(coins, amount, idx - 1, cache);
			return cache[idx][amount];
		}
		cache[idx][amount] = min(DFS_322(coins, amount, idx - 1, cache), DFS_322(coins, amount - coins[idx], idx, cache) + 1);
		return cache[idx][amount];
	}          

	int coinChange(vector<int>& coins, int amount) 
	{
		int n = (int)coins.size();
		vector<vector<int>> cache(n, vector<int>(amount + 1, -1));
		int ans = DFS_322(coins, amount, n - 1, cache);
		if (ans == 9999)
			return -1;
		return ans;
	}
	//TEST_322 end

	//TEST_2481 start
	int numberOfCuts(int n) 
	{
		if (n == 1)
			return 0;
		if (n & 1)
		{
			return n;
		}
		else
		{
			return n / 2;
		}
	}
	//TEST_2481 end

	//TEST_213 start
	int rob_213(vector<int>& nums) 
	{
		int n = (int)nums.size();
		if (n == 1)
			return nums[0];
		vector<int>dp(nums.size() + 2, 0);
		vector<int>dp2(nums.size() + 2, 0);
		for (int i = 0; i < (int)nums.size() - 1; i++)
		{
			dp[i + 2] = max(dp[i + 1], dp[i] + nums[i + 2]);
		}
		for (int i = 1; i < (int)nums.size(); i++)
		{
			dp2[i + 2] = max(dp2[i + 1], dp2[i] + nums[i + 2]);
		}
		return max(dp[n], dp2[n + 1]);
	}
	//TEST_213 end

	//TEST_2466 start
	int countGoodStrings(int low, int high, int zero, int one) 
	{
		long mod = 1000000007;
		long ans = 0;
		vector<int> dp(high + 1, 0);
		dp[0] = 1;
		for (int i = 0; i < high + 1; i++)
		{
			if (i >= zero)
			{
				dp[i] += dp[i - zero];
				dp[i] %= mod;
			}
			if (i >= one)
			{
				dp[i] += dp[i - one];
				dp[i] %= mod;
			}
			if (i >= low)
			{
				ans += dp[i];
				ans %= mod;
			}
		}
		return ans;

	}
	//TEST_2466 end

	//TEST_198 start
	//dp
	int rob_DP(vector<int>& nums)
	{
		int n = (int)nums.size();
		int dp1 = 0;
		int dp2 = 0;
		int dp3 = 0;
		for (int i = 0; i < n; i++)
		{
			dp3 = max(dp2, dp1 + nums[i]);
			dp1 = dp2;
			dp2 = dp3;
		}		
		return dp2;
	}
	//递归
	int DFS_198(int idx, vector<int> nums, int*& cache)
	{
		if (idx < 0)
			return 0;
		if (cache[idx] != -1)
			return cache[idx];

		cache[idx] = max(DFS_198(idx - 1, nums, cache), DFS_198(idx - 2, nums, cache) + nums[idx]);
		return max(DFS_198(idx - 1, nums, cache), DFS_198(idx - 2, nums, cache) + nums[idx]);
	}

	int rob(vector<int>& nums) 
	{
		int n = (int)nums.size();
		int ans = 0;
		int* cache = new int[n];
		for (int i = 0; i < n; i++)
			cache[i] = -1;
		return DFS_198(n - 1, nums, cache);
	}
	//TEST_198 end

	//TEST_494 start
	//dp
	int findTargetSumWays_DP(vector<int>& nums, int target) 
	{
		int sum = 0;
		for (int& num : nums) 
		{
			sum += num;
		}
		int diff = sum - target;
		if (diff < 0 || diff % 2 != 0) 
		{
			return 0;
		}
		int n = (int)nums.size(), neg = diff / 2;
		cout << neg << endl;
		vector<vector<int>> dp(n + 1, vector<int>(neg + 1));
		dp[0][0] = 1;
		for (int i = 1; i <= n; i++) 
		{
			int num = nums[i - 1];
			for (int j = 0; j <= neg; j++) 
			{
				dp[i][j] = dp[i - 1][j];
				if (j >= num) 
				{
					dp[i][j] += dp[i - 1][j - num];
				}
			}
		}
		for (int i = 0; i <= n; i++)
		{
			for (int j = 0; j <= neg; j++)
			{
				cout << dp[i][j] << " ";
			}
			cout << endl;
		}
		return dp[n][neg];
	}
	//Recursion
	int DFS_494(int i, int target, int length, vector<int>& nums)
	{
		if (i == length)
		{
			if (target == 0)
				return 1;
			else
				return 0;
		}
		return DFS_494(i + 1, target, length, nums) + DFS_494(i + 1, target - nums[i], length, nums);
	}

	int findTargetSumWays(vector<int>& nums, int target) 
	{
		int n = (int)nums.size();
		int sum = 0;
		for (int i : nums)
			sum += i;

		target += sum;
		if (target < 0 || target & 1)
			return 0;

		target /= 2;
		return DFS_494(0, target, n, nums);
	}
	//TEST_494 end

	//TEST_1177 start
	vector<bool> canMakePaliQueries(string s, vector<vector<int>>& queries) 
	{
		int n = (int)s.length(), q = (int)queries.size();
		int* sum = new int[n + 1] {};
		for (int i = 0; i < n; i++) 
		{
			int bit = 1 << (s[i] - 'a');
			sum[i + 1] = sum[i] ^ bit;
		}

		vector<bool> ans(q);
		for (int i = 0; i < q; i++) {
			auto& query = queries[i];
			int left = query[0], right = query[1], k = query[2];
			int m = __popcnt(sum[right + 1] ^ sum[left]);
			ans[i] = m / 2 <= k;
		}
		return ans;
	}
	//TEST_1177 end

	//TEST_1409 start
	vector<int> processQueries(vector<int>& queries, int m) 
	{
		vector<int> p(m);
		iota(p.begin(), p.end(), 1);
		vector<int> ans;
		for (int query : queries) {
			int pos = -1;
			for (int i = 0; i < m; ++i) {
				if (p[i] == query) {
					pos = i;
					break;
				}
			}
			ans.push_back(pos);
			p.erase(p.begin() + pos);
			p.insert(p.begin(), query);
		}
		return ans;
	}
	//TEST_1409 end

	//TEST_1395 start
	//树状数组初次尝试（抄一下）
	int numTeams_Treearray(vector<int>& rating)
	{
		const int N = (int)1e5 + 10;
		int n = (int)rating.size();
		int ans = 0;
		int* tr1 = new int[N] {};
		int* tr2 = new int[N] {};
		Double_Tree_Array stock;
		for (int i : rating)
			stock.update(tr2, i, 1);
		for (int i = 0; i < n; i++)
		{
			int t = rating[i];
			stock.update(tr2, t, -1);
			ans += stock.query(tr1, t - 1) * (stock.query(tr2, N - 1) - stock.query(tr2, t));
			ans += (stock.query(tr1, N - 1) - stock.query(tr1, t)) * stock.query(tr2, t - 1);
			stock.update(tr1, t, 1);
		}
		return ans;
	}
	//正常做法
	int numTeams(vector<int>& rating) 
	{
		int n = (int)rating.size();
		int ans = 0;
		for (int j = 1; j < n - 1; j++)
		{
			int iless = 0, imore = 0;
			int kless = 0, kmore = 0;
			for (int i = 0; i < j; i++)
			{
				if (rating[i] > rating[j])
					imore++;
				else if (rating[i] < rating[j])
					iless++;
			}
			for (int k = j + 1; k < n; k++)
			{
				if (rating[k] > rating[j])
					kmore++;
				else if (rating[k] < rating[j])
					kless++;
			}
			ans += imore * kless + iless * kmore;
		}
		return ans;
	}
	//TEST_1395 end

	//TEST_1375 start
	int numTimesAllBlue(vector<int>& flips) 
	{
		int ans = 0;
		int Max = INT_MIN;
		for (int i = 0; i < (int)flips.size(); i++)
		{
			Max = max(Max, flips[i]);
			if (Max == (i + 1))
				ans++;
		}
		return ans;
	}
	//TEST_1375 end

	//TEST_137 start
	//woshishabi
	int singleNumber(vector<int>& nums) 
	{
		int a = 0, b = 0;
		for (auto num : nums)
		{
			a = (a ^ num) & ~b;
			b = (b ^ num) & ~a;
		}
		return a;
	}
	//TEST_137 end

	//TEST_120 start
	//可以优化，因为每一行的状态只跟上一行有关，所以只用开辟2n的额外空间就可以
	int minimumTotal(vector<vector<int>>& triangle) 
	{
		int n = (int)triangle.size();
		vector<vector<int>> dp(n, vector<int>(n));
		dp[0][0] = triangle[0][0];

		for (int i = 1; i < n; i++)
		{
			for (int j = 0; j <= i; j++)
			{
				if (j == 0)
					dp[i][j] = dp[i - 1][j] + triangle[i][j];
				else if (j == i)
					dp[i][j] = dp[i - 1][j - 1] + triangle[i][j];
				else
				{
					dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j];
				}
			}
		}
		int ans = INT_MAX;

		for (int i = 0; i < n; i++)
		{
			ans = min(ans, dp[n - 1][i]);
		}
		return ans;
	}
	//TEST_120 end

	//TEST_130 start
	//BFS
	void solve_BFS(vector<vector<char>>& board)
	{
		queue<pair<int, int>> stock;
		int row = (int)board.size();
		int col = (int)board[0].size();
		for (int i = 0; i < row; i++)
		{
			if (board[i][0] == 'O')
				stock.push(make_pair(i, 0));
			if (board[i][col - 1] == 'O')
				stock.push(make_pair(i, col - 1));
		}
		for (int i = 0; i < col; i++)
		{
			if (board[0][i] == 'O')
				stock.push(make_pair(0, i));
			if (board[row - 1][i] == 'O')
				stock.push(make_pair(row - 1, i));
		}
		while (!stock.empty())
		{
			pair<int, int> a = stock.front();
			stock.pop();
			if (board[a.first][a.second] == 'O')
			{
				board[a.first][a.second] = '!';
				int x = a.first + 1;
				int y = a.second;
				if (!(x == row || y == col || x == -1 || y == -1))
					stock.push(make_pair(x, y));
				x = a.first - 1;
				y = a.second;
				if (!(x == row || y == col || x == -1 || y == -1))
					stock.push(make_pair(x, y));
				x = a.first;
				y = a.second + 1;
				if (!(x == row || y == col || x == -1 || y == -1))
					stock.push(make_pair(x, y));
				x = a.first;
				y = a.second - 1;
				if (!(x == row || y == col || x == -1 || y == -1))
					stock.push(make_pair(x, y));
			}
		}
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				if (board[i][j] == '!')
					board[i][j] = 'O';
				else if (board[i][j] == 'O')
					board[i][j] = 'X';
			}
		}
	}
	//DFS
	void DFS_130(int x, int y, int row, int col, vector<vector<char>>& board)
	{
		if (x == row || y == col || x == -1 || y == -1)
			return;
		if (board[x][y] == 'O')
		{
			board[x][y] = '!';
			DFS_130(x + 1, y, row, col, board);
			DFS_130(x - 1, y, row, col, board);
			DFS_130(x, y + 1, row, col, board);
			DFS_130(x, y - 1, row, col, board);
		}
	}

	void solve(vector<vector<char>>& board) 
	{
		int row = (int)board.size();
		int col = (int)board[0].size();
		for (int i = 0; i < row; i++)
		{
			if (board[i][0] == 'O')
				DFS_130(i, 0, row, col, board);
			if (board[i][col -1] == 'O')
				DFS_130(i, col - 1, row, col, board);
		}
		for (int i = 0; i < col; i++)
		{
			if (board[0][i] == 'O')
				DFS_130(0, i, row, col, board);
			if (board[row - 1][i] == 'O')
				DFS_130(row - 1, i, row, col, board);
		}
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				if (board[i][j] == '!')
					board[i][j] = 'O';
				else if (board[i][j] == 'O')
					board[i][j] = 'X';
			}
		}
	}
	//TEST_130 end

	//TEST_2475 start
	int unequalTriplets(vector<int>& nums) 
	{
		unordered_map<int, int> stock;
		for (int i : nums)
			stock[i]++;

		int n = (int)nums.size();
		int ans = 0, pre = 0;
		for (auto i = stock.begin(); i != stock.end(); i++)
		{
			int curr = i->second;
			ans += pre * curr * (n - pre - curr);
			pre += curr;
		}
		return ans;
	}
	//TEST_2475 end

	//TEST_128 start
	//Union_Find
	int longestConsecutive_UF(vector<int>& nums)
	{
		map<int, int> stock;
		UF uf((int)nums.size());

		for (int i = 0; i < (int)nums.size(); i++)
		{
			if (stock.count(nums[i]))
				continue;
			if (stock.count(nums[i] - 1))
				uf._union(i, stock[nums[i] - 1]);
			if (stock.count(nums[i] + 1))
				uf._union(i, stock[nums[i] + 1]);
			stock[nums[i]] = i;
		}
		return uf.getMaxConnectSize();
	}

	//hashmap
	int longestConsecutive(vector<int>& nums) 
	{
		int cur_num = 0;
		int cur_len = 0;
		int longest_len = 0;
		unordered_set<int> stock;
		for (int i : nums)
			stock.insert(i);
		
		for (int i : nums)
		{
			if (!stock.count(i - 1))
			{
				cur_len = 1;
				cur_num = i + 1;
				while (stock.count(cur_num))
				{
					cur_len++;
					cur_num++;
				}
				longest_len = max(longest_len, cur_len);
			}
		}
		return longest_len;
	}
	//TEST_128 end

	//TEST_80 start
	int removeDuplicates(vector<int>& nums) 
	{
		int n = (int)nums.size();
		if (n <= 2)
			return n;
		int slow = 2, fast = 2;
		while (fast < n)
		{
			if (nums[slow - 2] != nums[fast]) 
			{
				nums[slow] = nums[fast];
				++slow;
			}
			++fast;
		}
		return slow;
	}
	//TEST_80 end

	//TEST_90 start
	void DFS_90(vector<int>& combine, vector<vector<int>>& ans, int idx, vector<int> nums)
	{
		ans.push_back(combine);

		for (int i = idx; i < (int)nums.size(); i++)
		{
			if (i > idx && nums[i] == nums[i - 1])
				continue;
			combine.push_back(nums[i]);
			DFS_90(combine, ans, i + 1, nums);
			combine.pop_back();
		}
	}

	vector<vector<int>> subsetsWithDup(vector<int>& nums) 
	{
		vector<int> combine;
		vector<vector<int>> ans;
		sort(nums.begin(), nums.end());
		DFS_90(combine, ans, 0, nums);
		return ans;
	}
	//TEST_90 end

	//TEST_75 start
	void sortColors(vector<int>& nums) {
		int n = (int)nums.size();
		int p0 = 0, p2 = n - 1;
		for (int i = 0; i <= p2; ++i) {
			while (i <= p2 && nums[i] == 2) {
				swap(nums[i], nums[p2]);
				--p2;
			}
			if (nums[i] == 0) {
				swap(nums[i], nums[p0]);
				++p0;
			}
		}
	}
	//TEST_75 end

	//TEST_93 start
	void DFS_93(vector<string>& ans, int n, int pre, string& s)
	{
		if (pre >= s.length())
			return;
		if (n == 0 && (s.length() - pre) > 3)
			return;
		if (n == 0 && (s.length() - pre) > 1 && s[pre] == '0')
			return;
		if (n == 0 && (stoi(s.substr(pre)) > stoi("255")))
			return;
		if (n == 0)
		{
			ans.push_back(s);
			return;
		}
		for (int i = 1; i < 4; i++)
		{
			if (i == 1)
			{
				s.insert(pre + i, ".");
				DFS_93(ans, n - 1, pre + i + 1, s);
				s.erase(pre + i, 1);
			}
			else
			{
				if (s[pre] == '0')
					continue;
				if (i == 3 && (stoi(s.substr(pre, 3)) > stoi("255")))
					continue;
				if ((pre + i) >= s.length())
					continue;
				s.insert(pre + i, ".");
				DFS_93(ans, n - 1, pre + i + 1, s);
				s.erase(pre + i, 1);
			}
		}
	}

	vector<string> restoreIpAddresses(string s) 
	{
		vector<string> ans;
		string combine;
		DFS_93(ans, 3, 0, s);
		return ans;
	}
	//TEST_93 end

	//TEST_131 start
	void DFS_131(vector<string>& combine, vector<vector<string>>& ans, int start,int end, vector<vector<bool>>& dp, string s)
	{
		for (int i = start; i <= end; i++)
		{
			if (dp[start][i])
			{
				combine.push_back(s.substr(start, i - start + 1));
				if (i == s.length() - 1)
				{
					ans.push_back(combine);
					combine.pop_back();
					return;
				}
				DFS_131(combine, ans, i + 1, end, dp, s);
				combine.pop_back();
			}
		}
	}

	vector<vector<string>> partition(string s) 
	{
		vector<vector<bool>> dp(s.length(), vector<bool>(s.length()));

		for (size_t i = 0; i < s.length(); i++)
			dp[i][i] = true;
		for (size_t L = 2; L <= s.length(); L++)
		{
			for (size_t i = 0; i < s.length(); i++)
			{
				int j = (int)(i + L - 1);
				if (j >= s.length())
					break;
				if (s[i] != s[j])
					dp[i][j] = false;
				else
				{
					if (j - i < 3)
						dp[i][j] = true;
					else
						dp[i][j] = dp[i + 1][j - 1];
				}
			}
		}
		vector<string> combine;
		vector<vector<string>> ans;
		DFS_131(combine, ans, 0, (int)s.length() - 1, dp, s);
		return ans;
	}
	//TEST_131 end

	//TEST_171 start
	int titleToNumber(string columnTitle) 
	{
		long ans = 0;
		long multiply = 1;
		for (int i = (int)columnTitle.size() - 1; i >= 0; i--)
		{
			ans += (columnTitle[i] - 64) * multiply;
			multiply *= 26;
		}
		return ans;
	}
	//TEST_171 end

	//TEST_125 start
	//双指针解法
	bool isPalindrome_two_pointer(string s) {
		string sgood;
		for (char ch : s) {
			if (isalnum(ch)) {
				sgood += tolower(ch);
			}
		}
		int n = (int)sgood.size();
		int left = 0, right = n - 1;
		while (left < right) {
			if (sgood[left] != sgood[right]) {
				return false;
			}
			++left;
			--right;
		}
		return true;
	}
	//自己的解法我他妈像个智障！！！
	bool isPalindrome(string s) 
	{
		vector<char> str;
		for (size_t i = 0; i < s.length(); i++)
		{
			if ((s[i] >= 65 && s[i] <= 90) || (s[i] >= 97 && s[i] <= 122) || (s[i] >= 48 && s[i] <= 57))
				str.push_back(s[i]);
		}
		auto index = str.begin();
		for (int i = (int)s.length() - 1; i >= 0; i--)
		{
			if ((s[i] >= 65 && s[i] <= 90) || (s[i] >= 97 && s[i] <= 122) || (s[i] >= 48 && s[i] <= 57))
			{
				if (s[i] >= 65 && s[i] <= 90 && s[i] != *index && s[i] != *index - 32)
					return false;
				else if (s[i] >= 97 && s[i] <= 122 && s[i] != *index && s[i] != *index + 32)
					return false;
				else if (s[i] >= 48 && s[i] <= 57 && s[i] != *index)
					return false;
				index++;
			}
		}
		return true;
	}
	//TEST_125 end

	//TEST_2611 start
	//这个是贪心加排序
	int miceAndCheese_sort(vector<int>& reward1, vector<int>& reward2, int k) {
		int ans = 0;
		int n = (int)reward1.size();
		vector<int> diffs(n);
		for (int i = 0; i < n; i++) {
			ans += reward2[i];
			diffs[i] = reward1[i] - reward2[i];
		}
		sort(diffs.begin(), diffs.end());
		for (int i = 1; i <= k; i++) {
			ans += diffs[n - i];
		}
		return ans;
	}
	//这个是贪心加优先队列
	int miceAndCheese(vector<int>& reward1, vector<int>& reward2, int k) 
	{
		int ans = 0;
		if (k == 0)
		{
			for (int i : reward2)
				ans += i;
			return ans;
		}
		priority_queue<int, vector<int>, greater<int>> min_heap;
		for (int i = 0; i < k; i++)
		{
			min_heap.push(reward1[i] - reward2[i]);
			ans += reward1[i];
		}
		for (int i = k; i < (int)reward1.size(); i++)
		{
			int curr = reward1[i] - reward2[i];
			if (curr > min_heap.top())
			{
				ans = ans - min_heap.top() + reward1[i];
				min_heap.pop();
				min_heap.push(curr);
			}
			else
			{
				ans += reward2[i];
			}
		}
		return ans;

	}
	//TEST_2611 end

	//TEST_67 start
	string addBinary(string a, string b) 
	{
		string ans = "";
		int p1 = (int)a.length() - 1;
		int p2 = (int)b.length() - 1;
		int carry = 0;
		while (p1 >= 0 || p2 >= 0)
		{
			int curr;
			if (p1 < 0)
				curr = (b[p2] - '0') + carry;
			else if (p2 < 0)
				curr = (a[p1] - '0') + carry;
			else
				curr = (a[p1] - '0') + (b[p2] - '0') + carry;
			if (curr == 3)
			{
				carry = 1;
				ans.insert(ans.begin(), '1');
			}
			else if (curr == 2)
			{
				carry = 1;
				ans.insert(ans.begin(), '0');
			}
			else if(curr == 1)
			{
				carry = 0;
				ans.insert(ans.begin(), '1');
			}
			else
			{
				carry = 0;
				ans.insert(ans.begin(), '0');
			}
			p1--;
			p2--;
		}
		if (carry == 1)
			ans.insert(ans.begin(), '1');
		return ans;
	}
	//TEST_67 end

	//TEST_58 start
	int lengthOfLastWord(string s) 
	{
		int end = (int)s.length() - 1;
		while (s[end] == ' ') 
			end--;
		int begin = end;
		while (begin >= 0 && s[begin] != ' ' )
		{
			begin--;
		}
		return end - begin;
	}
	//TSET_58 end

	//TEST_49 start
	vector<vector<string>> groupAnagrams_sort(vector<string>& strs) 
	{
		unordered_map<string, vector<string>> mp;
		for (string& str : strs) {
			string key = str;
			sort(key.begin(), key.end());
			mp[key].emplace_back(str);
		}
		vector<vector<string>> ans;
		for (auto it = mp.begin(); it != mp.end(); ++it) {
			ans.emplace_back(it->second);
		}
		return ans;
	}
	//偷的别人的思想，试图给字母异位词一个特殊哈希值，但是此答案不是对的，并没有处理哈希碰撞，但是可以过leetcode
	long getId_49(string a)
	{
		long ans = 1;
		vector<long> map{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127};

		for (long ch : a)
		{
			ans *= map[ch - 'a'];
			ans %= 1073676287;
		}
		return ans;
	}

	vector<vector<string>> groupAnagrams(vector<string>& strs) 
	{
		map<long, vector<long>> stock;
		vector<vector<string>> ans;
		for (size_t i = 0; i < strs.size(); i++)
		{
			long curr = getId_49(strs[i]);
			stock[curr].push_back((int)i);
		}
		for (auto i = stock.begin(); i != stock.end(); i++)
		{
			vector<long> curr = i->second;
			vector<string> sub_ans;
			for (size_t j = 0; j < curr.size(); j++)
			{
				sub_ans.push_back(strs[curr[j]]);
			}
			ans.push_back(sub_ans);
		}

		return ans;
	}
	//TEST_49 end

	//TEST_48 start
	void rotate(vector<vector<int>>& matrix) 
	{
		int n = (int)matrix.size();
		for (int i = 0; i < n; i++)
		{
			for (int j = i + 1; j < n; j++)
			{
				int curr = matrix[i][j];
				matrix[i][j] = matrix[j][i];
				matrix[j][i] = curr;
			}
		}
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n / 2; j++)
			{
				int curr = matrix[i][j];
				matrix[i][j] = matrix[i][n - j - 1];
				matrix[i][n - j - 1] = curr;
			}
		}

	}
	//TEST_48 end

	//TEST_64 start
	int minPathSum(vector<vector<int>>& grid) 
	{
		int row = (int)grid.size();
		int col = (int)grid[0].size();
		vector<vector<int>> dp(row, vector<int>(col));
		dp[0][0] = grid[0][0];
		for (int i = 1; i < row; i++)
			dp[i][0] = dp[i - 1][0] + grid[i][0];
		for (int i = 1; i < col; i++)
			dp[0][i] = dp[0][i - 1] + grid[0][i];
		for (int i = 1; i < row; i++)
		{
			for (int j = 1; j < col; j++)
			{
				dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
			}
		}
		return dp[row - 1][col - 1]; 
	}
	//TEST_64 end

	//TEST_73 start
	void setZeroes(vector<vector<int>>& matrix) 
	{
		bool rowFlag = false;
		//判断首行
		for (size_t i = 0; i < matrix[0].size(); i++) {
			if (matrix[0][i] == 0) {
				rowFlag = true;
				break;
			}
		}

		bool colFlag = false;
		for (size_t i = 0; i < matrix.size(); i++) {
			if (matrix[i][0] == 0) {
				colFlag = true;
				break;
			}
		}

		for (size_t i = 1; i < matrix.size(); i++) {
			for (size_t j = 1; j < matrix[0].size(); j++) {
				if (matrix[i][j] == 0) {
					matrix[i][0] = 0;
					matrix[0][j] = 0;
				}
			}
		}

		for (size_t i = 1; i < matrix[0].size(); i++) {
			if (matrix[0][i] == 0) {
				for (size_t j = 0; j < matrix.size(); j++) {
					matrix[j][i] = 0;
				}
			}
		}

		for (size_t i = 1; i < matrix.size(); i++) {
			if (matrix[i][0] == 0) {
				for (size_t j = 0; j < matrix[0].size(); j++) {
					matrix[i][j] = 0;
				}
			}
		}
		if (rowFlag) {
			for (size_t i = 0; i < matrix[0].size(); i++) {
				matrix[0][i] = 0;
			}
		}
		if (colFlag) {
			for (size_t i = 0; i < matrix.size(); i++) {
				matrix[i][0] = 0;
			}
		}
	}
	//TEST_73 end

	//TEST_2352 start
	int equalPairs(vector<vector<int>>& grid) 
	{
		int ans = 0;
		map<vector<int>, int> stock;
		for (vector<int> i : grid)
			stock[i]++;
		for (size_t i = 0; i < grid.size(); i++)
		{
			vector<int> temp;
			for (size_t j = 0; j < grid.size(); j++)
			{
				temp.push_back(grid[j][i]);
			}
			ans += stock[temp];
		}
		return ans;
	}
	//TEST_2352 end

	//TEST_621 start
	int leastInterval(vector<char>& tasks, int n) 
	{
		int sameFreq = 0;
		int maxkind = 0;
		int a[26]{};
		for (char ch : tasks)
			a[ch - 'A']++;
		for (int i : a)
		{
			if (i > maxkind)
				maxkind = i;
		}
		for (int i : a)
			if (i == maxkind)
				sameFreq++;
		int preserve = (maxkind - 1) * (n + 1) + sameFreq;

		return max((int)tasks.size(), preserve);
	}
	//TEST_621 end

	//TEST_561 start
	int arrayPairSum(vector<int>& nums) 
	{
		int ans = 0;
		sort(nums.begin(), nums.end());
		size_t i = 0;
		for (; i < nums.size() - 1; i += 2)
		{
			ans += min(nums[i], nums[i + 1]);
		}
		return ans;
	}
	//TEST_561 end

	//TEST_406 start
	vector<vector<int>> reconstructQueue(vector<vector<int>>& people) 
	{
		auto c = [](const vector<int>& a, const vector<int>& b)
		{
			if (a[0] == b[0]) return a[1] < b[1];
			return a[0] > b[0];
		};
		sort(people.begin(), people.end(), c);
		//为什么用list是因为vector的底层实现是数组，list的底层实现是列表
		//很明显，链表更适合不断的插入，数组不适合，因为数组需要扩容
		list<vector<int>> ans;
		for (size_t i = 0; i < people.size(); i++)
		{
			int pos = people[i][1];
			auto j = ans.begin();
			while (pos--)
			{
				j++;
			}
			ans.insert(j, people[i]);
		}
		return vector<vector<int>>(ans.begin(), ans.end());
	}
	//TEST_406 end

	//TEST_2460 start
	vector<int> applyOperations(vector<int>& nums) 
	{
		int n = (int)nums.size();
		for (int i = 0, j = 0; i < n; i++) {
			if (i + 1 < n && nums[i] == nums[i + 1]) {
				nums[i] *= 2;
				nums[i + 1] = 0;
			}
			if (nums[i] != 0) {
				swap(nums[i], nums[j]);
				j++;
			}
		}
		return nums;
	}
	//TEST_2460 end

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
	//最大的最小，最小的最大可以用二分法
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
	//要做剪枝，否则不行
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