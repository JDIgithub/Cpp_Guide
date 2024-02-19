
#include <iostream>
#include <vector>
<<<<<<< HEAD
#include <algorithm>
#include "class.h"
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <list>
#include <thread>
#include <functional>



void threadFunction(){
  std::cout << "Im thread function" << std::endl;

}

int main()
{
  std::thread(threadFunction);  // !! Program will actually crash because when we are going out of this scope the thread object will be destroyed
}                               // But the thread is still running !!  
=======




#include <iostream>
#include <vector>

// Iterative version
int binarySearch(const std::vector<int>& array, int target) {
  
  int low = 0;
  int high = array.size() - 1;

  while (low <= high) {
    int mid = low + (high - low) / 2;

    if (array[mid] == target) {
      return mid;
    } else if (array[mid] < target) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
    return -1; // Target not found
}
>>>>>>> ca7a28dacfa471687debe8ab293e44f3eea055a4

int main() {
  
  std::vector<int> array = {2, 3, 4, 10, 40};
  int target = 10;
  int result = binarySearch(array, target);
    
  if (result != -1) {
    std::cout << "Element found at index " << result << std::endl;
  } else {
    std::cout << "Element not found in the array" << std::endl;
  }

  return 0;
}





