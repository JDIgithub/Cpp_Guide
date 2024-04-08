#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
/*
Problem: Merge Intervals
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.

Constraints:
1 <= intervals.length <= 10^4
intervals[i].length == 2
0 <= starti <= endi <= 10^4
Approach:
A common approach to solve this problem is as follows:

Sort the intervals by their start times. This makes sure that we only need to check the current interval with the last merged interval.
Initialize an empty list to hold the merged intervals.
Iterate through the sorted intervals and for each one:
If the list of merged intervals is empty or if the current interval does not overlap with the last merged interval, simply add the current interval to the list of merged intervals.
If the current interval does overlap with the last merged interval, merge them by updating the end time of the last merged interval to the maximum end time between both intervals.
Return the list of merged intervals.

*/

std::vector<std::pair<int,int>> mergeIntervals(std::vector<std::pair<int,int>> &intervals){

  if(intervals.empty()) return {};

  std::sort(intervals.begin(),intervals.end());

  std::vector<std::pair<int, int>> merged;
  for (const auto& interval : intervals) {
    // If the list of merged intervals is empty or if the current interval does not overlap
    if (merged.empty() || merged.back().second < interval.first) {
      merged.push_back(interval);
    } else {
      // There is an overlap, so we merge the current interval with the previous one
      merged.back().second = std::max(merged.back().second, interval.second);
    }
  }

  return merged;
}


int main1() {

  std::vector<std::pair<int,int>> intervals {{1,3},{2,6},{8,10},{15,18}};
  for(const auto &pair: mergeIntervals(intervals)){
    std::cout << "{" << pair.first << "," << pair.second << "} ";
  }
  return 0;
}
//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------

/*
Problem Statement: Product of Array Except Self
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

Example:
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Explanation:
By calculating the product of all numbers except nums[i] without using division, we look for an efficient way to do this in O(n) time.

Approach:
Left and Right Products List: For each element, we could try to calculate the product of all the elements to the left and all the elements to the right. We can then multiply these two values to get our desired product for each position.
Initialize two arrays, left and right, of the same length as nums, where left[i] contains the product of all elements to the left of i, and right[i] contains the product of all elements to the right of i.
The final answer for each position i can be calculated as left[i] * right[i].
We can optimize space by using the answer array to first calculate the left products, then compute the right products on the fly while updating the answer array.
Let's implement this approach in C++:

*/

std::vector<int> productExceptSelf(const std::vector<int> &nums) {
  
  std::vector<int> answer(nums.size(),1);

  int left = 1;
  for(int i = 0; i < nums.size(); i++){
    answer[i] = left;
    left *= nums[i];
  }
  int right = 1;
  for(int i = (nums.size()-1); i >= 0; i--){
    answer[i] *= right;   // Multiple the left size with the right size to get the result
    right *= nums[i];
  }

  return answer;
}

int main2() {

  std::vector<int> nums = {1,2,3,4};

  for(int product: productExceptSelf(nums)){
    std::cout << product << " ";
  }

  return 0;
}
//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------

/*
Problem: Longest Substring Without Repeating Characters
Given a string s, find the length of the longest substring without repeating characters.

Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

Constraints:
0 <= s.length <= 5 * 10^4
s consists of English letters, digits, symbols, and spaces.
Approach:
A common approach to solve this problem is using the Sliding Window technique combined with a Hash Map to track characters and their indices in the string.
This allows us to skip characters immediately when we encounter a repeated character.

*/

int longestSubstring(const std::string& str){

  if(str.empty()) return 0;

  std::unordered_map<char,int> charIndx;
  int maxLength = 0; // Store the maximum length found.
  int start = 0; // Sliding window start index.
  
  for(int end = 0; end< str.size(); end++){

    if(charIndx.find(str[end]) != charIndx.end() && charIndx[str[end]] >= start){ 
      start = charIndx[str[end]] + 1; // Move the start of the window past this character's last occurrence.
    }
    charIndx[str[end]] = end; // Saves characeters and their idex into hash map. If char is already there just update his last index
    maxLength = std::max(maxLength, end - start + 1); // If length of actual windows is higher than maxlength then update it
  }

  return maxLength;
}

int main3() {

  std::string str {"pwwkew"};

  std::cout << longestSubstring(str);

  return 0;
}


//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------

/*

Problem: Coin Change
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Write a function to compute the fewest number of coins that you need to make up that amount. 
If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

Example 1:
Input: coins = [1, 2, 5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:
Input: coins = [2], amount = 3
Output: -1

Example 3:
Input: coins = [1], amount = 0
Output: 0

Constraints:
1 <= coins.length <= 12
1 <= coins[i] <= 2^31 - 1
0 <= amount <= 10^4
Approach:
This problem is a classic example of dynamic programming. Here's a high-level approach to solve this problem:

Initialization: 

Create an array dp of size amount + 1 to store the minimum number of coins needed for each amount from 0 to amount. 
Initialize dp[0] to 0 because no coins are needed for the amount 0. All other dp[i] should be initialized to a value 
that represents infinity (since we are looking for the minimum number of coins, initializing to amount + 1 is a practical approach as you cannot have 
more coins than amount itself if each coin were of denomination 1).

Filling the dp Table: 

For each amount i from 1 to amount, and for each coin value coin in coins, if coin <= i, update dp[i] as the minimum of dp[i] and dp[i - coin] + 1. 
This step essentially checks if using the current coin would result in a smaller number of coins to make up the current amount i.

Result: 
After filling in the dp table, dp[amount] will hold the minimum number of coins needed to make up the amount. 
If dp[amount] is still its initialized value (amount + 1), it means it's impossible to make up the amount with the given coins, and you should return -1.

*/

int coinChange(const std::vector<int> &coins, int amount){

  std::vector<int> dp(amount + 1);
  dp[0] = 0;
  for(int i = 1; i < dp.size(); i++){
    dp[i] = amount + 1;
    for(int coin: coins){
      if(coin <= i){
        dp[i] = std::min(dp[i],dp[i-coin]+1);
      }
    }
  }

  return dp[amount] > amount ? -1 : dp[amount];
}

int main4() {


  std::cout << coinChange({1,2,5},11) << std::endl;
  std::cout << coinChange({2},3) << std::endl;
  std::cout << coinChange({1},0) << std::endl;



  return 0;
}


//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------

/*
Problem: Edit Distance
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character
Example 1:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation:
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation:
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

Constraints:
0 <= word1.length, word2.length <= 500
word1 and word2 consist of lowercase English letters.
Approach:
This is a classic problem that can be solved using dynamic programming. The idea is to build a 2D array dp where dp[i][j] represents the minimum 
number of operations required to convert the first i characters of word1 to the first j characters of word2.

*/

int minDistance(std::string word1, std::string word2) {

  int m = word1.length(), n = word2.length();
  std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

  // Initialize DP table first row and first column
  for (int i = 0; i <= m; i++) dp[i][0] = i;
  for (int j = 0; j <= n; j++) dp[0][j] = j;



  for(int i = 1; i<= m; i++){

    for(int j = 1; j <= n; j++){

      if(word1[i-1] == word2[j-1]){
        dp[i][j] = dp[i-1][j-1];
      } else {                    // Deletion   // Insertion    // Replacement
        dp[i][j] = 1 + std::min( {dp[i-1][j],   dp[i][j-1],     dp[i - 1][j - 1] });

      }



    }

  }
  return dp[m][n];
}



int main5() {

  minDistance("horse","ros");



  return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------

/*

Problem: Maximum Subarray Sum
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A subarray is a contiguous part of an array.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum = 6.

Example 2:
Input: nums = [1]
Output: 1

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23

Constraints:
1 <= nums.length <= 10^5
-10^4 <= nums[i] <= 10^4
Approach: Kadane’s Algorithm
This problem can be efficiently solved using Kadane’s Algorithm, which is a dynamic programming approach. 
The core idea of Kadane’s Algorithm is to scan through the entire array and at each position find the maximum sum of the subarray ending there.
This is done by keeping a running count of the maximum sum ending at the previous position and using this to calculate the maximum sum ending at the current position.

Implementation Steps:
Initialize two integer variables, say maxSoFar and maxEndingHere, with the first element of the array. maxSoFar will hold the final maximum subarray sum
and maxEndingHere will hold the maximum sum of the subarray ending at the current position as we iterate through the array.
Iterate through the array starting from the second element (since we used the first element to initialize our variables) and for each element, update maxEndingHere. If maxEndingHere becomes less than the current element, reset maxEndingHere to the value of the current element.
Update maxSoFar if maxEndingHere is greater than maxSoFar.
Return maxSoFar as the largest sum of the contiguous subarray.

*/

int maxSubArray(const std::vector<int> &nums){

  int maxSoFar = nums[0];
  int maxEndingHere = nums[0];

  for(int i = 1; i < nums.size(); i++){
    maxEndingHere = std::max(maxEndingHere + nums[i],nums[i]);    
    maxSoFar = std::max(maxSoFar,maxEndingHere);
  }

  return maxSoFar;
}

int main6() {

  std::cout << maxSubArray({-2,1,-3,4,-1,2,1,-5,4}) << std::endl;
  std::cout << maxSubArray({1}) << std::endl;
  std::cout << maxSubArray({5,4,-1,7,8}) << std::endl;


  return 0;
}


//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------

/*

Problem: Implement a Thread-Safe Queue
Implement a thread-safe queue that can be used in a producer-consumer scenario. 
The queue should be able to support multiple producers and consumers using threads. 
The key operations to implement are enqueue (to add an item to the back of the queue) and dequeue (to remove an item from the front of the queue).
If the queue is empty, the dequeue operation should wait until an item is available. Use mutexes and condition variables to achieve thread safety and synchronization.

Requirements:
Thread-Safe Enqueue: Allows one or more producer threads to add items to the queue.
Thread-Safe Dequeue: Allows one or more consumer threads to remove items from the queue. 
If the queue is empty, the consumer thread should wait for an item to be enqueued.

Synchronization: 
Ensure that access to the queue is properly synchronized between threads to prevent race conditions.
Condition Variables: 
Use condition variables to signal state changes in the queue, specifically for notifying consumer threads when new items are added to an empty queue.
Suggested Implementation Steps:
Define a template class ThreadSafeQueue that encapsulates a standard queue (e.g., std::queue) for storage.
Use a mutex (e.g., std::mutex) to protect access to the underlying queue.
Use condition variables (e.g., std::condition_variable) to manage waiting and signaling between producer and consumer threads.
Implement the enqueue method to add an item to the queue. After adding an item, notify a waiting consumer thread.
Implement the dequeue method to remove and return an item from the front of the queue. If the queue is empty, wait for an item to be enqueued.



*/


template<typename T>
class ThreadSafeQueue{

private:

  std::queue<T> queue;
  mutable std::mutex mutex; // why mutable 
  std::condition_variable condVar;

public:

  void enqueue(T item){
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(std::move(item));
    condVar.notify_one(); // Notify one waiting consumer
  }

  T dequeue(){
    std::unique_lock<std::mutex> lock(mutex);
    condVar.wait(lock, [this] { return !queue.empty(); }); // Wait until the queue is not empty
    T value = std::move(queue.front());
    queue.pop();
    return value;
  }

};


int main7() {

  ThreadSafeQueue<int> queue;
  std::thread producer( [&queue](){
    for(int i = 0; i < 10;i++){
      std::cout << "Producing " << i << std::endl;
      queue.enqueue(i);
    }
  });

  std::thread consumer( [&queue](){
    for(int i = 0; i < 10;i++){
      int item = queue.dequeue();
      std::cout << "Consuming " << item << std::endl;      
    }
  });

  producer.join();
  consumer.join();

  return 0;
}
#include <unordered_map>


//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------
// Dynamic Programming

static int counter = 0;

std::vector<int> memo(100,-1);
std::unordered_map<int,int> memoMap;

int fib(int n){
  counter++;
  if(memo[n] != -1){
    return memo[n];
  }

  if (n == 0 || n == 1){
    return n;
  }

  memo[n] = fib(n - 1) + fib(n - 2);
  return memo[n];
}


int fibBottomUp(int n){
  std::vector<int> fibList;
  fibList.push_back(0);
  fibList.push_back(1);

  for(int index = 2; index <= n; index++){
    fibList[index] = fibList[index - 1] + fibList[index - 2];
  }
  return fibList[n];
  
}

int main8() {
  
  int n = 20;

  std::cout << "\nFib of " << n << " = " << fib(n);
  std::cout << "\nCounter: " << counter << std::endl;   // O(2n - 1) -> 2*20 - 1 = 39   

  return 0;
}


