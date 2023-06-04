#pragma once
#include <iostream>
#include <fstream>
#include <string>
template <class T>
class List;

template <class T>
class ListNode
{
private:
	T _item;
	ListNode<T> *_next;

public:
	ListNode(T);
	T content() { return _item; };
	// void print() { cout << _item; };
	friend class List<T>;
};

template <class T>
class List
{
private:
	int _size;
	ListNode<T>* _head;
	ListNode<T>* _current;

public:

	// for the following functions, you cannot assume that the list is not empty or empty...
	List() { _size = 0;	_head = NULL; _current = NULL; };

	~List();
	void insertHead(T);
	void print(bool withNL = false); // print the items in one single row if false, otherwise, print each item in a new line
	int size() { return _size; };
	bool exist(T);
	bool empty() { return _size == 0; };


	// .. except for the following functions, we can assume that the list is not empty when you call them.
	T headItem();
	void removeHead();

	// Simple Iterator
	void start() { _current = _head; };
	T current() { if (_current) return _current->_item; };
	void next() { if (_current) _current = _current->_next; };
	bool end() { return _current == NULL;};
		
};

#include "simpleLinkedListTemplate.cpp"
