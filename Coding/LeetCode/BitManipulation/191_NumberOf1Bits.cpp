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


using namespace std::literals;

/* 191. Numbers of 1 Bits

Write a function that takes the binary representation of a positive integer and returns the number of set bits it has (also known as the Hamming weight).

Example 1:

  Input: n = 11
  Output: 3

Explanation:

  The input binary string 1011 has a total of three set bits.

Example 2:

  Input: n = 128
  Output: 1

Explanation:

  The input binary string 10000000 has a total of one set bit.

Example 3:

  Input: n = 2147483645

  Output: 30

Explanation:

  The input binary string 1111111111111111111111111111101 has a total of thirty set bits.

Constraints:

1 <= n <= 231 - 1
 

Follow up: If this function is called many times, how would you optimize it?

*/

int hammingWeight(int n) {
  int count = 0;
  while (n != 0) {
    if((n & 1)){ 
      count++; 
    }
    n = n >> 1; // Shifts n to the right by 1 position
  }
  return count;
}


// Brian Kernighan Algorithm
int hammingWeightBrianKernighan(int n) {
    int count = 0;
    while (n != 0) {
        n = n & (n - 1); // Clear the least significant bit set to 1
        // It takes that many steps to get to 0 as there are bits set to 1 in the number
        count++;
    }
    return count;
}



// Look Up map
/* 
static int lookupTable[256] = {0};

// Function to initialize the lookup table
void initializeLookupTable() {
    for (int i = 0; i < 256; i++) {
        lookupTable[i] = (i & 1) + lookupTable[i >> 1];
    }
}

// Function to compute hamming weight using the lookup table
int hammingWeight(int n) {
    initializeLookupTable(); // Ensure this is called once, ideally during program initialization
    return lookupTable[n & 0xFF] + 
           lookupTable[(n >> 8) & 0xFF] + 
           lookupTable[(n >> 16) & 0xFF] + 
           lookupTable[(n >> 24) & 0xFF];
}
*/
int main(){

  auto xx = hammingWeight(11);
  std::cout << xx;
  return 0;
}