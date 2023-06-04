// this is a functor
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
class add_x {
public:
	add_x(int val) : x(val) {}  // Constructor
	int operator()(int y) const { return x + y; }

private:
	int x;
};

int main()
{
	// Now you can use it like this:
	add_x add42(42); // create an instance of the functor class
	int i = add42(8); // and "call" it
	assert(i == 50); // and it added 42 to its argument

	std::vector<int> in; // assume this contains a bunch of values)
	in.push_back(3);
	in.push_back(6);
	std::vector<int> out(in.size());
	// Pass a functor to std::transform, which calls the functor on every element 
	// in the input sequence, and stores the result to the output sequence
	std::transform(in.begin(), in.end(), out.begin(), add_x(1));
	assert(out[3] == in[3] + 1); // for all i
}