#include <iostream>
#include "SortQ.h"

using namespace std;

int main()
{
	SortQ<int> test;
	test.enQueue(4);
	test.enQueue(1);
	test.enQueue(9);
	test.enQueue(6);
	test.enQueue(7);
	test.enQueue(12);
	test.enQueue(0);
	test.enQueue(-3);
	test.enQueue(-6);

	test.print_two_order();

	test.deQueue();
	test.print_two_order();

	test.deQueue();
	test.print_two_order();

	test.deQueue();
	test.print_two_order();

	test.deQueue();
	test.print_two_order();
	test.deQueue();
	test.print_two_order();
	test.deQueue();
	test.print_two_order();
	test.deQueue();

	test.print_two_order();
	test.deQueue();

	test.print_two_order();
	test.deQueue();

	test.print_two_order();
	test.deQueue();

	test.print_two_order();
	test.print_two_order();
	test.print_two_order();
	test.print_two_order();
	test.deQueue();
	test.deQueue();
	test.deQueue();
	test.deQueue();
	test.enQueue(7);
	test.enQueue(12);
	test.enQueue(0);
	test.enQueue(-3);
	test.enQueue(-6);
	test.print_two_order();
}