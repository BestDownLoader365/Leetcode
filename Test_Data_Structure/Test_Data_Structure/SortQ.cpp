#pragma once

#ifndef HII
#define HII

#include <iostream>
#include "SortQ.h"
using namespace std;

template <class T>
Node<T>::Node(T n)
{
	_item = n;
	_Qnext = NULL;
	_Snext = NULL;
	_Sprev = NULL;
}

template <class T>
SortQ<T>::SortQ()
{
	_Qhead = NULL;
	_Qrear = NULL;
	_Shead = NULL;
	_Srear = NULL;
	_size = 0;
}

template <class T>
void SortQ<T>::enQueue(T x)
{
	
	Node<T>* t1 = new Node<T>(x);
	if (!_Qhead)
	{
		_Qhead = t1;
		_Qrear = t1;
		_Shead = t1;
		_Srear = t1;
		++_size;
		return;
	}

	_Qrear->_Qnext = t1;
	_Qrear = t1;

	if (_Shead->_item > x)
	{
			Node<T>* t2 = _Shead;
			_Shead = t1;
			t1->_Snext = t2;
			t2->_Sprev = t1;
			++_size;
			return;
	}
	if(_Srear->_item < x)
	{
			_Srear->_Snext = t1;
			t1->_Sprev = _Srear;
			_Srear = t1;
			++_size;
			return;
	}

	Node<T>* current = _Shead;
	while (!(current->_item < x && current->_Snext->_item > x))
	{
		current = current->_Snext;
	}
	Node<T>* t3 = current->_Snext;
	current->_Snext = t1;
	t1->_Sprev = current;
	t1->_Snext = t3;
	t3->_Sprev = t1;
	++_size;
}

template <class T>
void SortQ<T>::deQueue()
{
	if (_size == 0)
	{
		cout << "Empty Queue!!!" << endl;
		return;
	}
	Node<T>* t1 = _Qhead;
	if (_size == 1)
	{
		delete _Qhead;
		_Qhead = NULL;
		_Qrear = NULL;
		_Shead = NULL;
		_Srear = NULL;
		--_size;
		return;
	}
	_Qhead = _Qhead->_Qnext;

	if (_Shead == t1)
	{
		_Shead = _Shead->_Snext;
		_Shead->_Sprev = NULL;

	}
	else if (_Srear == t1)
	{
		_Srear = _Srear->_Sprev;
		_Srear->_Snext = NULL;

	}
	else
	{
		Node<T>* t2 = t1->_Sprev;
		Node<T>* t3 = t1->_Snext;
		t2->_Snext = t3;
		t3->_Sprev = t2;

	}

	delete t1;

	--_size;
}

template <class T>
void SortQ<T>::print_two_order()
{
	if (_size == 0)
	{
		cout << "Empty Queue!!!" << endl;
		return;
	}
	cout << "Queue Order Print: ";

	Node<T>* t1 = _Qhead;
	while (t1)
	{
		cout << t1->_item << " ";
		t1 = t1->_Qnext;
	}
	cout << endl;

	cout << "Sorted List Print: ";

	Node<T>* t2 = _Shead;
	while (t2)
	{
		cout << t2->_item << " ";
		t2 = t2->_Snext;
	}
	cout << endl;
}

template <class T>
SortQ<T>::~SortQ()
{
	while (_Qhead)
	{
		Node<T>* temp = _Qhead;
		_Qhead = _Qhead->_Qnext;
		delete temp;
	}
}

#endif