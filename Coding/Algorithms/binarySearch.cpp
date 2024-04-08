
#include <thread>
#include <iostream>

int binarySearch(int array[],int left, int right, int x){
  
  if (right >= left){
    int mid = left + (right - left)/2;  // (right + left)/2 is possible but this version helps prevent overflow

    if(array[mid] == x) { return mid; } 
    if(array[mid] > x) { 
      return binarySearch(array, left, mid - 1, x); // If element is higher we need to search in the left half
    } else {
      return binarySearch(array, mid + 1, right, x); // If element is lower we need to search in the right half
    }

  } else {
    return -1;
  }
}

int main()
{
  int array[] = { 2, 5, 8, 12, 16, 23, 38, 56, 72, 91 };
  int n = sizeof(array)/sizeof(array[0]);
  int x = 23;
  int result = binarySearch(array, 0, n - 1, x);

  if(result != -1){
    std::cout << "Element " << x <<  " is present at index " << result << std::endl;
  } else {
    std::cout << "Element " << x <<  " is not present" << std::endl;
  }
}