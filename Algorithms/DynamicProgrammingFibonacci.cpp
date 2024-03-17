#include <iostream>
#include <vector>
#include <unordered_map>


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

int main() {
  
  int n = 20;

  std::cout << "\nFib of " << n << " = " << fib(n);
  std::cout << "\nCounter: " << counter << std::endl;   // O(2n - 1) -> 2*20 - 1 = 39   

  return 0;
}