
#include <iostream>
#include <vector>

void mergeSort(std::vector<int>& array, int left, int right);
void merge(std::vector<int>& array, int left, int middle, int right);

int main() {
  
  std::vector<int> array = {12, 11, 13, 5, 6, 7};
  mergeSort(array, 0, array.size() - 1);
  std::cout << "\nSorted array is \n";
  for(int i : array) { std::cout << i << " ";}
  std::cout << std::endl;

  return 0;
}

// Takes array and cuts it in half
// It is called recursively till we have only single element
// Then calls merge() to merge them back together
// So we will get sorted sub-arrays twice bigger every time merge will be called till we have the full sorted array
void mergeSort(std::vector<int>& array, int leftIndex, int rightIndex) {
  if (leftIndex < rightIndex) {
    return;
  }
  // Find the middle point
  int middleIndex = leftIndex + (rightIndex - leftIndex) / 2;

  // Recursive calls for left and right halves
  mergeSort(array, leftIndex, middleIndex);
  mergeSort(array, middleIndex + 1, rightIndex);

  // Merge the sorted halves
  merge(array, leftIndex, middleIndex, rightIndex);
}

// Takes two sorted arrays and combines them into one sorted array
// Well technically we pass just one array but it has 2 sorted sub-arrays inside
// And we copy them to two separated arrays inside the function
void merge(std::vector<int>& array, int leftIndex, int middleIndex, int rightIndex) {
  // Compute sizes of the two sub-arrays 
  int leftArraySize = middleIndex - leftIndex + 1;
  int rightArraySize = rightIndex - middleIndex;
  // Create temp arrays
  std::vector<int> Left(leftArraySize), Right(rightArraySize);
  // Copy data to temp arrays L[] and R[]
  for (int i = 0; i < leftArraySize; i++){ Left[i] = array[leftIndex + i]; }
  for (int j = 0; j < rightArraySize; j++){ Right[j] = array[middleIndex + 1 + j]; }
  
  // Merge the temp arrays back into the original array
  int index = leftIndex;
  int i = 0, j = 0;
  // Runs until one of the sub-arrays is fully iterated through
  while (i < leftArraySize && j < rightArraySize) {
    if (Left[i] <= Right[j]) {
      array[index] = Left[i];
      i++;
    } else {
      array[index] = Right[j];
      j++;
    }
    index++;
  }
  // Because the while loop above is broken when only one of the array is done we need to finish the second one:
  // Copy the remaining elements of Left[], if there are any
  while (i < leftArraySize) {
    array[index] = Left[i];
    i++;
    index++;
  }
  // Copy the remaining elements of Right[], if there are any
  while (j < rightArraySize) {
    array[index] = Right[j];
    j++;
    index++;
  }
}