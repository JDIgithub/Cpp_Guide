#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

// Coding exercise Vector: 1 ---------------------------------------------------------------------------------------------------------------------------

	//   | Description:                                        |
	//   | - This function removes all occurrences of a        |
	//   |   given value ('val') from an integer vector.       |
	//   | - It modifies the input vector 'nums'. 

  //   | Return type: void                                   |
	//   |                                                     |
	//   | Tips:                                               |
	//   | - Uses two pointers 'i' and 'j' for traversal.      |
	//   | - 'i' points to the last element that is not 'val'. |
	//   | - 'j' is used for iterating over the array.         |
	
void removeElement (std::vector<int>& nums, int val) {

  int i{0};
  for(int j {0}; j< nums.size(); j++){
    if(nums[j]!=val){
      nums[i] = nums[j];
      i++;
    } 
  }
  nums.resize(i);

// Or this does the same
//  auto newEnd = std::remove_if(nums.begin(),nums.end(), [val](int num){ return num ==val; });
// nums.erase(newEnd, nums.end());

}

// Coding exercise Vector: 2 ---------------------------------------------------------------------------------------------------------------------------

//   | Description:                                        |
//   | - This function finds the maximum and minimum       |
//   |   values in a given list of integers.               |
//   | - It uses a single loop to go through the list.     |
//   |                                                     |
//   | Return type: vector<int>                            |
//   | - Returns a vector containing maximum and minimum.  |
//   |                                                     |
//   | Tips:                                               |
//   | - 'maximum' and 'minimum' keep track of the         |
//   |   highest and lowest values found.                  |

std::vector<int> findMaxMin(std::vector<int> & myList){

  std::vector<int> MaxMin {0,0};
  if(!myList.empty()){
    MaxMin[0] = myList[0];
    MaxMin[1] = myList[0];
    for(int num: myList){
      if(num > MaxMin[0]){
        MaxMin[0] = num;
      }
      if(num < MaxMin[1]){
        MaxMin[1] = num;
      }
    }

  }
  return MaxMin;

}



// Coding exercise Vector: 3 ---------------------------------------------------------------------------------------------------------------------------
	//   | Description:                                        |
	//   | - This function finds the longest string in         |
	//   |   a given list of strings.                          |
	//   | - It uses a single loop to check the length         |
	//   |   of each string.                                   |
	//   |                                                     |
	//   | Return type: string                                 |
	//   | - Returns the longest string found in the list.     |
	//   |                                                     |
	//   | Tips:                                               |
	//   | - 'longestString' keeps track of the longest        |
	//   |   string found so far.                              |

std::string findLongestString(std::vector<std::string>& stringList) {

  std::string longest{};
  for(std::string word: stringList){
    if(word.size() > longest.size()){
      longest = word;
    }
  }
  return longest;
}


// Coding exercise Vector: 4 ---------------------------------------------------------------------------------------------------------------------------

	//   | Description:                                        |
	//   | - This function removes duplicate integers          |
	//   |   from a sorted array in-place.                     |
	//   | - It uses two pointers to achieve this.             |
	//   |                                                     |
	//   | Return type: int                                    |
	//   | - Returns the length of the new array.              |
	//   |                                                     |
	//   | Tips:                                               |
	//   | - 'writePointer' is used for storing unique values. |
	//   | - 'readPointer' is used for reading array values.   |

int removeDuplicates(std::vector<int>& nums) {

  if(nums.empty()){ return 0;}

  int writePointer {1};
 

  for( int readPointer{1}; readPointer < nums.size(); readPointer++){

    if(nums[readPointer] != nums[readPointer - 1]){
      nums[writePointer] = nums[readPointer];
      writePointer++;
    } 
  }
  return writePointer;
}


// Coding exercise Vector: 5 ---------------------------------------------------------------------------------------------------------------------------
	//   | Description:                                        |
	//   | - This function calculates the maximum profit       |
	//   |   from buying and selling a stock.                  |
	//   | - It iterates through the 'prices' array once.      |
	//   |                                                     |
	//   | Return type: int                                    |
	//   |                                                     |
	//   | Tips:                                               |
	//   | - 'minPrice' keeps track of the lowest price.       |
	//   | - 'maxProfit' stores the maximum profit found.      |
	//   | - Use 'min' and 'max' functions to update values.   |

int maxProfit(std:: vector<int>& prices) {

  int minPrice = prices[0];
  int maxProfit{0};

  for(int price: prices){
    minPrice = std::min(minPrice,price);
    maxProfit = std::max(maxProfit, price - minPrice);
  }  
  return maxProfit;
}



// Coding exercise Vector: 6 ---------------------------------------------------------------------------------------------------------------------------
	//   | Description:                                        |
	//   | - This function rotates the array 'nums' to the     |
	//   |   right by 'k' steps.                               |
	//   | - The method used involves reversing parts of the   |
	//   |   array.                                            |
	//   |                                                     |
	//   | Return type: void                                   |
	//   |                                                     |
	//   | Tips:                                               |
	//   | - 'k' is first made smaller by taking modulo size.  |
	//   | - Three reversals are done to achieve the rotation. |
  
  // 1 2 3 4 5 6 7   shift by k = 3
  // We can shift it by first reversing sub-array that start at first element 
  // and ends at element that will be at the end of the array when everything is done sums[nums.size()-k-1]
  // 4 3 2 1 5 6 7
  // Then we reverse sub-array of the elements that we did not touch yet so start at element[nums.size()-k]
  // and ends at the last element of the array
  // 4 3 2 1 7 6 5
  // Then we reverse the whole array and the result is original array shifted by k positions to the right
  // 5 6 7 1 2 3 4

void rotate(std::vector<int>& nums, int k) {

  if(nums.empty() || k == 0) {return;}
  
  
  k = k % nums.size();
  // Reverse the first part
  for(int start{0}, end{nums.size() - k - 1}; start < end; start++,end--){
    std::swap(nums[start],nums[end]);
  }

  // Reverse the second part
  for (int start = nums.size() - k, end = nums.size() - 1; start < end; start++, end--) {
    std::swap(nums[start],nums[end]);
  }

  // Reverse the whole array
  for (int start = 0, end = nums.size() - 1; start < end; start++, end--) {
    std::swap(nums[start],nums[end]);
  }
}


// Coding exercise Vector: 7 ---------------------------------------------------------------------------------------------------------------------------
	//   | Description:                                        |
	//   | - This function finds the sum of the contiguous     |
	//   |   subarray with the largest sum from the given      |
	//   |   array 'nums'.                                     |
	//   | - It uses Kadane's algorithm for this task.         |
	//   |                                                     |
	//   | Return type: int                                    |
	//   |                                                     |
	//   | Tips:                                               |
	//   | - 'maxSum' stores the maximum subarray sum.         |
	//   | - 'currentSum' keeps track of the sum of the        |
	//   |   subarray ending at the current index.             |
	//   | - Use 'max' to update 'maxSum' and 'currentSum'.    |

int maxSubarray(std::vector<int>& nums) {


}

int main(){

  std::vector<int> intVec {1,2,4,4,5,7};
  //removeElement(intVec,3);

  removeDuplicates(intVec);


  return 0;
}