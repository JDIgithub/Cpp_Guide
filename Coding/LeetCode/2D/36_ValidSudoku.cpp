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
#include <csignal>
#include <optional>

using namespace std;


/* 36 Valid Sudoku

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

  Each row must contain the digits 1-9 without repetition.
  Each column must contain the digits 1-9 without repetition.
  Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

Note:

  A Sudoku board (partially filled) could be valid but is not necessarily solvable.
  Only the filled cells need to be validated according to the mentioned rules.
 

Example 1:


Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
Example 2:

Input: board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
 

Constraints:

board.length == 9
board[i].length == 9
board[i][j] is a digit 1-9 or '.'.


*/


bool isValidSudoku(std::vector<std::vector<char>>& board) {

  std::vector<std::vector<int>> colCheck(9, std::vector<int>(9, 0));
  std::vector<std::vector<int>> miniBoxCheck(9, std::vector<int>(9, 0));

  for(int row = 0; row < board[0].size(); row++){

    std::vector<int> rowCheck(9,0);
    for(int col = 0; col < board.size(); col++){

      if(board[row][col] == '.') continue; 
      int num = board[row][col] - '1';      // 1 -> 0 so it starts with index 0

      if(rowCheck[num]++ > 0) return false; // It will check if this position is occupied 
      //                                       and then occupies it with incrementing from 0 to 1 or from 1 to 2 -> returns false
      if(colCheck[col][num]++ > 0) return false;
      //               0/3/6          0/1/2   
      int boxIndex = (row / 3) * 3 + col / 3;
      if(miniBoxCheck[boxIndex][num]++ > 0) return false;

    }
  }
  return true;
}




int main() {


  std::vector<std::vector<char>> board = { 
    {'5','3','.','.','7','.','.','.','.'},
    {'6','.','3','1','9','5','.','.','.'},
    {'.','9','8','.','.','.','.','6','.'},
    {'8','.','.','.','6','.','.','.','3'},
    {'4','.','.','8','.','3','.','.','1'},
    {'7','.','.','.','2','.','.','.','6'},
    {'.','6','.','.','.','.','2','8','.'},
    {'.','.','.','4','1','9','.','.','5'},
    {'.','.','.','.','8','.','.','7','9'}
  };


  auto xx = isValidSudoku(board);

  std::cout << xx;



  return 0;


}





