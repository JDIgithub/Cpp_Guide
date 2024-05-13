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

// 861. Scope After Flipping Matrix

/*

You are given an m x n binary matrix grid.
A move consists of choosing any row or column and toggling each value in that row or column (i.e., changing all 0's to 1's, and all 1's to 0's).

Every row of the matrix is interpreted as a binary number, and the score of the matrix is the sum of these numbers.

Return the highest possible score after making any number of moves (including zero moves).

 

Example 1:


Input: grid = [[0,0,1,1],[1,0,1,0],[1,1,0,0]]
Output: 39
Explanation: 0b1111 + 0b1001 + 0b1111 = 15 + 9 + 15 = 39
Example 2:

Input: grid = [[0]]
Output: 1
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 20
grid[i][j] is either 0 or 1.
*/

void invertRows(std::vector<std::vector<int>>& grid,std::vector<int> &oneCounter){

  bool rowStays{false};
  for(int row = 0; row < grid.size(); row++){
    
    rowStays = grid[row][0];

    for(int col = 0; col < grid[row].size(); col++){
      if(!rowStays){
        grid[row][col] = 1 - grid[row][col];
      }
      oneCounter[col] += grid[row][col];
    }
  }
}

void invertColumns(std::vector<std::vector<int>>& grid, std::vector<int> &oneCounter){

  for(int col = 0; col < grid[0].size(); col++){ 
    if(oneCounter[col] > grid.size()/2){
      for(int row = 0; row < grid.size(); row++){
        grid[row][col] = 1 - grid[row][col];
      }
    }
  }
}

int matrixScore(std::vector<std::vector<int>>& grid) {

  std::vector<int> oneCounter(grid[0].size(),0);
  invertRows(grid,oneCounter);
  invertColumns(grid,oneCounter);

  uint32_t binaryNumber = 0;

  for(int row = 0; row < grid.size(); row++){
    for(int col = 0; col < grid[row].size(); col++){
      binaryNumber += grid[row][col] << (grid[row].size() - col - 1);
    }
  }

  return binaryNumber;
}



int main(){

  std::vector<std::vector<int>> grid = {{0,1,1},{1,1,1},{0,1,0}};

  std::cout << matrixScore(grid) << std::endl;



  return 0;
}


