
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

void mergeSort(std::vector<int>& array, int left, int right) {
  if (left < right) {
    // Find the middle point
    int middle = left + (right - left) / 2;

    // Sort first and second halves
    mergeSort(array, left, middle);
    mergeSort(array, middle + 1, right);

    // Merge the sorted halves
    merge(array, left, middle, right);
  }
}

void merge(std::vector<int>& array, int left, int middle, int right) {
   
  int n1 = middle - left + 1;
  int n2 = right - middle;

  // Create temp arrays
  std::vector<int> Left(n1), Right(n2);

  // Copy data to temp arrays L[] and R[]
  for (int i = 0; i < n1; i++){ Left[i] = array[left + i]; }
  for (int j = 0; j < n2; j++){ Right[j] = array[middle + 1 + j]; }
  
  // Merge the temp arrays back into arr[l..r]
  int i = 0, j = 0, k = left;
  while (i < n1 && j < n2) {
    if (Left[i] <= Right[j]) {
      array[k] = Left[i];
      i++;
    } else {
      array[k] = Right[j];
      j++;
    }
    k++;
  }

  // Copy the remaining elements of L[], if there are any
  while (i < n1) {
    array[k] = Left[i];
    i++;
    k++;
  }

  // Copy the remaining elements of R[], if there are any
  while (j < n2) {
    array[k] = Right[j];
    j++;
    k++;
  }
}
