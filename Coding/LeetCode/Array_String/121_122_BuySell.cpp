#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>
#include <list>
#include <random>
#include <atomic>

// 121. Best Time to Buy and Sell

/*

You are given an array prices where prices[i] is the price of a given stock on the i-th day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.


Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
 

Constraints:

1 <= prices.length <= 105
0 <= prices[i] <= 104
  
*/


int maxProfit(std::vector<int>& prices) {
  
  if(prices.empty()) return 0;

  int buy = prices[0];
  int maxProfit = 0;
  
  for(int i = 0; i < prices.size(); i++){

    buy = std::min(buy, prices[i]);
    maxProfit = std::max(maxProfit, (prices[i]-buy)); 

  }


  return maxProfit;
}

// 122. Best Time to Buy and Sell 2

/*

You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. 
However, you can buy it then immediately sell it on the same day.
Find and return the maximum profit you can achieve.

 
Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Total profit is 4.
Example 3:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: There is no way to make a positive profit, so we never buy the stock to achieve the maximum profit of 0.
 

Constraints:

1 <= prices.length <= 3 * 104
0 <= prices[i] <= 104
  
*/


int maxProfit2(std::vector<int>& prices) {
  
  if(prices.empty() || prices.size() == 1) return 0;
  
  int buy = -1;
  int maxProfit = 0;
  
  for(int currentDay = 0; currentDay < prices.size() - 1; currentDay++){

    if((prices[currentDay+1] - prices[currentDay]) > 0){
      if(buy == -1){
        buy = prices[currentDay];
      }
    } else {
      if(buy != -1){
        maxProfit += (prices[currentDay] - buy); 
        buy = -1;
      }
    } 
  }

  if(buy != -1){
    maxProfit += (prices[prices.size()-1] - buy); 
  }

  return maxProfit;
}

int main(){

  std::vector<int> prices = {1,2,3,4,5};

  std::cout << maxProfit2(prices);

  return 0;
}




