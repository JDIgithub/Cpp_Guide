#include <string>
#include <algorithm>
#include <thread>
#include <iostream>
#include <functional>
#include <stack>
#include <queue>


void showStack(std::stack<int> stck){

  while(!stck.empty()){
    std::cout << '\t' << stck.top();
    stck.pop();
  }
  std::cout << '\n';

}

void showQ(std::queue<int> que){

  while(!que.empty()){
    std::cout << '\t' << que.front();
    que.pop();
  }
  std::cout << '\n';


}



bool areBracketsBalanced(const std::string& expr){

  std::stack<char> stck;

  for(char c: expr){
    // if char is opening bracket just add it to the stack and go for next char  
    if(c == '(' || c == '[' || c == '{'){
      stck.push(c);
      continue;
    }
    // if we are here char is not opening bracket -> control if it is closing bracket
    switch (c){
      case ')':
        if(stck.top() != '(') { return false;}
        stck.pop();
        break;  
      case '}':
         if(stck.top() != '{') { return false;}
        stck.pop();
        break; 
      case ']':
        if(stck.top() != '[') { return false;}
        stck.pop();
        break;
      default:
        break;   // if the character is not bracket, just go for next character
    }
  }

  return true;
}

int main() {

  std::string expr {"{(jojo)}[]"};

  if (areBracketsBalanced(expr)){
    std::cout << "Balanced " << std::endl;
  } else {
    std::cout << "Not Balanced " << std::endl;
  }
}