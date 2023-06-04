#pragma once

template <class T>
class SortQ;

template <class T>
class Node
{
private:
	Node<T>* _Snext;
	Node<T>* _Sprev;
	Node<T>* _Qnext;
	T _item;

public:
	Node(T);
	friend class SortQ<T>;
};

template <class T>
class SortQ 
{
private:
	Node<T>* _Qhead;
	Node<T>* _Qrear;
	Node<T>* _Shead;
	Node<T>* _Srear;
	int _size;

public:
	void enQueue(T);
	void deQueue();
	~SortQ();
	SortQ();
	void print_two_order();
};

#include "SortQ.cpp"