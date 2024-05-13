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

// 9. Palindrome Number

/*

Given an integer x, return true if x is a 
palindrome
, and false otherwise.

 

Example 1:

Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.
Example 2:

Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
Example 3:

Input: x = 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
 

Constraints:

-231 <= x <= 231 - 1
 

Follow up: Could you solve it without converting the integer to a string?
*/

bool isPalindrome(int x) {

  if (x < 0) return false;  // Negative numbers are not palindromes
  if (x < 10) return true;  // Single digit numbers are always palindromes

  long reversed = 0;  // Reversed must be able to store bigger number than the original -> long 
  int original = x;
  
  while (x > 0) {
    if(INT_MAX/reversed < 10){ return false;}
    reversed = reversed * 10 + x % 10;
    x /= 10;
  }

  return original == reversed;
}



int main(){

  std::cout << isPalindrome(1234567899) << std::endl;  

  return 0;
}


