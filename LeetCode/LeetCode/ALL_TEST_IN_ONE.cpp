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
using namespace std;
class Solution {

public:
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
                int buckett = bucket[j];
                if (buckett != 0)
                {
                    int cur = 0;
                    int vatt = vat[j];
                    while (vatt / buckett > i)
                    {
                        cur++;
                        buckett++;
                    }
                    sum += cur;
                }
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
    //Ä¦¶û¼ÆÊý·¨£¡£¡£¡
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