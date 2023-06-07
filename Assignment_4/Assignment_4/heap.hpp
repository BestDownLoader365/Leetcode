#pragma once

#include <iostream>
#include <math.h>
using namespace std;

#ifndef HEAPHPP
#define HEAPHPP

template <class T>
void Heap<T>::_bubbleUp(int index) {
	int up_index = (index - 1) / 2;
	if (up_index == -1)
	{
		return;
	}
	T temp = _heap[up_index];
	if (_heap[index] > temp)
	{
		_heap[up_index] = _heap[index];
		_heap[index] = temp;
		_bubbleUp(up_index);
	}
	else
	{
		return;
	}
}

template <class T>
void Heap<T>::_bubbleDown(int index) {
	
	int left_index = (index * 2) + 1;
	int right_index = (index * 2) + 2;

	if (left_index >= _n)
	{
		return;
	}
	else if (_heap[left_index] > _heap[right_index])
	{
		_bubbleUp(left_index);
		_bubbleDown(left_index);
	}
	else
	{
		_bubbleUp(right_index);
		_bubbleDown(right_index);
	}
}

template <class T>
void Heap<T>::insert(T item) {
	_heap[_n] = item;
	_bubbleUp(_n);
	++_n;
}

template <class T>
T Heap<T>::extractMax() {
	T max = _heap[0];
	_heap[0] = _heap[_n - 1];
	--_n;

	_bubbleDown(0);
	return max;
}


template <class T>
void Heap<T>::printHeapArray() {
	for (int i = 0; i < _n; i++)
		cout << _heap[i] << " ";
	cout << endl;
}

template <class T>
int Heap<T>::_lookFor(T x){ // not a very good implementation, but just use this for now.
    int i;
    for(i=0;i<_n;i++)
        if (_heap[i] == x)
            return i;
    
    return -1;
}

template <class T>
void Heap<T>::decreaseKey(T from, T to)
{
	int index = _lookFor(from);
	if (index == -1)
	{
		return;
	}
	else
	{
		_heap[index] = to;
		_bubbleDown(index);
	}
}


template <class T>
void Heap<T>::increaseKey(T from, T to)
{
	int index = _lookFor(from);
	if (index == -1)
	{
		return;
	}
	else
	{
		_heap[index] = to;
		_bubbleUp(index);
	}
}

template <class T>
void Heap<T>::deleteItem(T x)
{
	int index = _lookFor(x);
	if (index == -1)
		return;

	T delete_item = _heap[index];
	T last_item = _heap[_n - 1];
	_heap[index] = last_item;
	--_n;

	if (delete_item > last_item)
	{
		_bubbleDown(index);
	}
	else
	{
		_bubbleUp(index);
	}
}

template <class T>
void Heap<T>::printTree() {
    int parity = 0;
	if (_n == 0)
		return;
	int space = pow(2,1 + (int) log2f(_n)),i;
	int nLevel = (int) log2f(_n)+1;
	int index = 0,endIndex;
    int tempIndex;
	
	for (int l = 0; l < nLevel; l++)
	{
		index = 1;
        parity = 0;
		for (i = 0; i < l; i++)
			index *= 2;
		endIndex = index * 2 - 1;
		index--;
        tempIndex = index;
        while (index < _n && index < endIndex) {
            for (i = 0; i < space-1; i++)
                cout << " ";
            if(index==0)
                cout << "|";
            else if (parity)
                cout << "\\";
            else
                cout << "/";
            parity = !parity;
            for (i = 0; i < space; i++)
                cout << " ";
			index++;
		}
        cout << endl;
        index=tempIndex;
		while (index < _n && index < endIndex) {
			for (i = 0; i < (space-1-((int) log10(_heap[index]))); i++)
				cout << " ";
			cout << _heap[index];
			for (i = 0; i < space; i++)
				cout << " ";
			index++;
		}
		
		cout << endl;
		space /= 2;
	}

}






#endif
