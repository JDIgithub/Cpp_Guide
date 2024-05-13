#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>


using namespace std;

// 2373. Largest Local Values in a Matrix

/*

You are given an n x n integer matrix grid.
Generate an integer matrix maxLocal of size (n - 2) x (n - 2) such that:

  maxLocal[i][j] is equal to the largest value of the 3 x 3 matrix in grid centered around row i + 1 and column j + 1.

In other words, we want to find the largest value in every contiguous 3 x 3 matrix in grid.

Return the generated matrix.

 

Example 1:


Input: grid = [[9,9,8,1],[5,6,2,6],[8,2,6,4],[6,2,2,2]]
Output: [[9,9],[8,6]]
Explanation: The diagram above shows the original matrix and the generated matrix.
Notice that each value in the generated matrix corresponds to the largest value of a contiguous 3 x 3 matrix in grid.
Example 2:


Input: grid = [[1,1,1,1,1],[1,1,1,1,1],[1,1,2,1,1],[1,1,1,1,1],[1,1,1,1,1]]
Output: [[2,2,2],[2,2,2],[2,2,2]]
Explanation: Notice that the 2 is contained within every contiguous 3 x 3 matrix in grid.
 

Constraints:

n == grid.length == grid[i].length
3 <= n <= 100
1 <= grid[i][j] <= 100

*/

/*
vector<int> plusOne(vector<int>& digits) {
    

  for(int i = digits.size()-1; i >= 0 ; --i){  

    if(digits[i] < 9){

      digits[i]++;
      break;

    } else {
      
        digits.push_back(0);
      



      
    }  
  }
}
*/


volatile int counter = 0;

void task() {
  for(int i = 0; i < 100'000; ++i){
    ++counter;
  }
}



int main(){




  std::atomic<Test*> ptest = nullptr;
  Test *ptr = ptest;
  ptr->func();





  std::vector<std::thread> tasks;  
  for(int i = 0; i < 10; ++i){
    tasks.push_back(std::thread(task));
  }
  for(auto& task: tasks){
    task.join();
  }

  // Should be 1'000'000 but it is around 340'000 because volatile is not working in threading
  std::cout << counter << '\n';   

  return 0;
}


