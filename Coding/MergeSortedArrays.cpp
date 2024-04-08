#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
using namespace std;

// LeetCode 88. Merge Sorted Arrays

// Without STL
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    
  if(n == 0)return;
  int len1 = nums1.size();
  int end_idx = len1-1;
  while(n > 0 && m > 0){
    if(nums2[n-1] >= nums1[m-1]){
      nums1[end_idx] = nums2[n-1];
      n--;
    }else{
      nums1[end_idx] = nums1[m-1];
      m--;
    }
    end_idx--;
  }
  while (n > 0) {
    nums1[end_idx] = nums2[n-1];
    n--;
    end_idx--;
  }    
}

// With STL sort
void merge2(vector<int>& nums1, int m, vector<int>& nums2, int n) {
  
  for (int j = 0, i = m; j<n; j++){
    nums1[i] = nums2[j];
    i++;
  }
  sort(nums1.begin(),nums1.end());
}

// With priority_queue... but we are losing O(1) space complexity
void merge3(vector<int>& nums1, int m, vector<int>& nums2, int n) {
  
  auto compare = [](int l1, int l2) { return l1 > l2; };
  std::priority_queue<int, std::vector<int>, decltype(compare)> pq(compare);
  for(int i = 0; i < m; i++){
    pq.push(nums1[i]);
  }
  for(int num: nums2){
    pq.push(num);
  }  
  for(int i = 0; i < nums1.size(); i++){
    nums1[i] = pq.top();
    pq.pop();
  }
}


int main(){

  std::vector<int> nums1 {0};
  std::vector<int> nums2 {1};

  merge(nums1,0,nums2,1);

  return 0;
}


