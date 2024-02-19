
#include <iostream>
#include <vector>




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





