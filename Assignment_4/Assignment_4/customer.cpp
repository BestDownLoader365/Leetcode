#include "customer.h"
#include "heap.h"
#include <iostream>
using namespace std;
int comparisonWay = 0; // 0 for arrival time, 1 for processing time


bool Customer::operator>(const Customer& c) {
	return comparisonWay ? processing_time < c.processing_time : arrival_time < c.arrival_time; // a customer is "greater" if his time is shorter
};



void customerQueueTest(int n_cust) {
	int current_time = 0;
	int totalWaitingTime = 0;
	int currentWaitingTime = 0;
	int count_for_loop = 0;
	int i;
	int WT = 0; // waiting time for each customer
	Heap<Customer> queue;


	Customer* custList = new Customer[n_cust];
	cout << endl << endl << "Setup" << endl;
	cout << "List of customers and their arrival and processing times" << endl;
	for (i = 0; i < n_cust; i++)
	{
		custList[i].setAT(i);
		//custList[i].setPT((n_cust - i) % (n_cust / 2) + 1 + (i % 2) * (n_cust / 2));
		custList[i].setPT(i + 1);
		cout << "Customer " << i << " will arrive at time " << custList[i].AT() << " with PT=" << custList[i].PT() << endl;
	}
	cout << endl;

	for (int round = 0; round < 2; round++) {

			cout << endl << endl;
			cout << "Test Round " << round + 1 << endl;
			cout << (round == 0 ? "First come first serve" : "Customer with the least PT served first") << endl;
			comparisonWay = round;
			current_time = 0;
			count_for_loop = 0;
			totalWaitingTime = 0;
			currentWaitingTime = 0;
			bool first_time = true;
			while (first_time || !queue.empty())
			{
				first_time = false;
				for (int current_Customer = 0; current_Customer < n_cust; current_Customer += 1)
				{
					count_for_loop += 1;
					if (count_for_loop <= n_cust)
					{
						queue.insert(custList[current_Customer]);
						cout << "Customer arrives at time " << custList[current_Customer].AT() << " with PT=" << custList[current_Customer].PT() << endl;
					}
					if (currentWaitingTime == 0)
					{
						Customer c = queue.extractMax();
						currentWaitingTime += c.PT();
						if (current_time == c.AT())
						{
							cout << "Customer arrives when no one is waiting" << endl;
						}
						WT = (current_time - c.AT());
						cout << "Customer is served at time " << current_time << " with AT=" << c.AT() << " and PT=" << c.PT() << " after waiting for "
							<< WT << " min." << endl;

						totalWaitingTime += WT;
					}
					currentWaitingTime -= 1;
					current_time += 1;
				}
			}
			cout << "Total Waiting Time: " << totalWaitingTime << endl;
			cout << "Average Waiting Time: " << totalWaitingTime / (float)n_cust << endl;
		}
		delete[] custList;
	}

