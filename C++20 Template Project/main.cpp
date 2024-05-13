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

// 67. Add Binary

/*

Given two binary strings a and b, return their sum as a binary string.

Example 1:

Input: a = "11", b = "1"
Output: "100"
Example 2:

Input: a = "1010", b = "1011"
Output: "10101"

*/

string addBinary(string a, string b) {
  string ans;
  int carry = 0;
  int i = a.length() - 1;
  int j = b.length() - 1;
  while (i >= 0 || j >= 0 || carry) {
    if (i >= 0)
      carry += a[i--] - '0';
    if (j >= 0)
      carry += b[j--] - '0';
    ans += carry % 2 + '0';
    carry /= 2;
  }
  ans=std::to_string(carry%2)+ans;
 // std::reverse(begin(ans), end(ans));
  return ans;
}


std::string addBinary(std::string a, std::string b) {
  

  // Fill the shorter string with 0 to match the length of the longer string  10010,11 -> 10010,00011 
  while(a.size() < b.size()) {
    a = "0" + a;
  }
  while(b.size() < a.size()) {
    b = "0" + b;
  }

  int carry = 0;
  std::string result = "";

  for(int i = b.size()-1; i >= 0 ; --i)
  {    
    if(b[i] == '1' && a[i]=='1') {
      if(carry == 0) {
        result = "0" + result;
      } else {
        result = "1" + result;
      }
      carry = 1;
    } else if(b[i] =='0' && a[i] =='0') {
      if(carry == 0) {
        result = "0" + result;
      } else {
        result = "1" + result;
        carry = 0;
      }

    } else if((b[i]=='0' && a[i]=='1') || (b[i]=='1' && a[i] == '0')) { 
      if(carry == 0) { 
        result = "1" + result;
      } else { 
        result = "0" + result;     
      }
    }     
  }
  
  
  if(carry == 1) result = "1" + result;
  
  return result;
}


int main(){

  std::string s {"11"};
  std::string t {"1"};

  std::cout << addBinary(s,t);

  return 0;
}


