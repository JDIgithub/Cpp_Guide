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

using namespace std;

// 202. Happy Number
/*

Write an algorithm to determine if a number n is happy.
A happy number is a number defined by the following process:

  Starting with any positive integer, replace the number by the sum of the squares of its digits.
  Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
  Those numbers for which this process ends in 1 are happy.

Return true if n is a happy number, and false if not.
*/

int nextNumber(int n){
  int newNumber = 0;
    while(n!=0){
      int num = n%10;
      newNumber += num*num;
       n = n/10;
    }
  return newNumber;
}

bool isHappy(int n) {

  std::unordered_set<int> set;
  while(n!=1 && !set.count(n)){ // When we reach 1 or we reach to the same number twice -> end the loop
    set.insert(n);
    n = nextNumber(n);
  }
  return n==1;  // If n = 1 return true or false otherwise
}

int main(){

  
  auto result = isHappy(19);
  int pause = 0;
  
  return 0;
}






