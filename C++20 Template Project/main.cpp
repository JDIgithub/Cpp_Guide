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

// 1219. Path with Maximum Gold

/*

In a gold mine grid of size m x n, each cell in this mine has an integer representing the amount of gold in that cell, 0 if it is empty.
Return the maximum amount of gold you can collect under the conditions:

  Every time you are located in a cell you will collect all the gold in that cell.
  From your position, you can walk one step to the left, right, up, or down.
  You can't visit the same cell more than once.
  Never visit a cell with 0 gold.
  You can start and stop collecting gold from any position in the grid that has some gold.
 

Example 1:

  Input: grid = [[0,6,0],[5,8,7],[0,9,0]]
  Output: 24
  Explanation:
    [[0,6,0],
    [5,8,7],
    [0,9,0]]
    Path to get the maximum gold, 9 -> 8 -> 7.

Example 2:

  Input: grid = [[1,0,7],[2,0,6],[3,4,5],[0,3,0],[9,0,20]]
  Output: 28
  Explanation:
    [[1,0,7],
    [2,0,6],
    [3,4,5],
    [0,3,0],
    [9,0,20]]
    Path to get the maximum gold, 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7.
 


Constraints:

  m == grid.length
  n == grid[i].length
  1 <= m, n <= 15
  0 <= grid[i][j] <= 100
  There are at most 25 cells containing gold.

*/

// We should use memoization to store the results because we will need them more than one time when different starting position will copy the same path
int MyfindBestPath(vector<vector<int>>& grid,int row, int col){

  if(row >= grid.size() || row < 0) return 0;
  if(col >= grid[row].size() || col < 0) return 0;
  if(grid[row][col]==0) return 0;

  // Store the current value and mark the cell as visited by setting it to 0
  int current = grid[row][col];
  grid[row][col] = 0;

  // Calculate the best path from the current cell
  int collected = current + std::max({
    MyfindBestPath(grid, row + 1, col), // Down
    MyfindBestPath(grid, row - 1, col), // Up
    MyfindBestPath(grid, row, col + 1), // Right
    MyfindBestPath(grid, row, col - 1)  // Left
  });

  // Setting the original value back for the next starting positions
  grid[row][col] = current;

  return collected;
}


int MygetMaximumGold(vector<vector<int>>& grid) {
        
  int maxNum = 0;      
  for(int row = 0; row < grid.size(); row++){
    for(int col = 0; col < grid[row].size(); col++) {
      maxNum = std::max(maxNum,MyfindBestPath(grid,row,col));
    }
  }
  return maxNum;
}


// Better Solution ToDo

int row, col;
int gold=0;
    
    
void f(int i, int j, int sum,  vector<vector<int>>& grid){
  if (i<0 ||i>=row|| j<0 || j>=col||grid[i][j]==0) return;
  int tmp=grid[i][j];
  sum+=tmp;
  gold=max(gold, sum);
  grid[i][j]=0;
  f(i+1, j, sum, grid);
  f(i-1, j, sum, grid);
  f(i, j+1, sum, grid);
  f(i, j-1, sum, grid);
  //sum-=tmp;
  grid[i][j]=tmp;//backtracking
}

int getMaximumGold(vector<vector<int>>& grid) {
  row=grid.size(), col=grid[0].size();
  for(int i=0; i<row; i++)
    for(int j=0; j<col; j++){
      if (grid[i][j]!=0){
        f(i, j, 0, grid);
      }
    }
  return gold;
}

int main(){

  std::vector<std::vector<int>> grid = {{1,0,7},{2,0,6},{3,4,5},{0,3,0},{9,0,20}};

  std::cout << getMaximumGold(grid) << std::endl;



  return 0;
}



