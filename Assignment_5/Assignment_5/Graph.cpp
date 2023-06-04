#include "Graph.h"
//uncomment this to include your own "heap.h"
//we will assume that you use the same code in your previous assignment
#include "heap.h"

std::ostream& operator<<(std::ostream& os, nodeWeightPair const& n) {
	return os << " (idx:" << n._node << " w:" << n._weight << ")";
}


Graph::Graph(int n)
{
	_al = new List<nodeWeightPair> [n];
	_nv = n;
}

int Graph::shortestDistance(int s, int d)
{
	if (_nv == 1)
	{
		return -1;
	}

	Heap<nodeWeightPair> temp;
	int* hashmap = new int[_nv];
	int* parentIndex = new int[_nv];
	bool* visited = new bool[_nv];
	for (int i = 0; i < _nv; i += 1)
	{
		visited[i] = false;
		parentIndex[i] = -999999;
		hashmap[i] = -999999;
	}

	nodeWeightPair curr_release(s, 0);
	temp.insert(curr_release);
	hashmap[s] = 0;
	while (curr_release.nodeIndex() != d && !temp.empty())
	{
		curr_release = temp.extractMax();
		int index = curr_release.nodeIndex();
		if (!visited[index])
		{
			visited[index] = true;
			for (_al[index].start(); !_al[index].end(); _al[index].next())
			{
				int nodeIndex = _al[index].current().nodeIndex();
				temp.insert(nodeWeightPair(nodeIndex, -999999));
	
				int new_weight = curr_release.weight() - _al[index].current().weight();		
				if (new_weight > hashmap[nodeIndex])
				{
					temp.insert(nodeWeightPair(nodeIndex, new_weight));
					hashmap[nodeIndex] = new_weight;
					parentIndex[nodeIndex] = curr_release.nodeIndex();
				}
			}
		}
	}
	if (curr_release.nodeIndex() != d)
	{
		delete[] hashmap;
		delete[] parentIndex;
		delete[] visited;
		return -1;
	}
	else
	{
		cout << "Path: ";
		
		int current = d;
		int* output = new int[_nv];
		int count = 0;
		for (int i = 0; i < _nv; i += 1)
		{
			output[i] = -1;
		}
		while (current != s)
		{
			output[count] = current;
			count += 1;
			current = parentIndex[current];
		}
			cout << s << " ";
		for (int i = count - 1; i >= 0; i -= 1)
		{
			if (i != 0)
				cout << output[i] << " ";
			else
				cout << output[i];
		}

		cout << endl;
		delete[] parentIndex;
		delete[] hashmap;
		delete[] output;
		delete[] visited;
		return -curr_release.weight();
	}
}
void Graph::addEdge(int s, int d, int w)
{
	_al[s].insertHead(nodeWeightPair(d, w));
}

void Graph::printGraph()
{
	for (int i=0; i < _nv; i++)
	{
		cout << "Node " << i << ": ";
		for (_al[i].start(); !_al[i].end(); _al[i].next())
			cout << " (" << _al[i].current().nodeIndex() << "," << _al[i].current().weight() << ")";
		cout << endl;

	}

}
Graph::~Graph()
{

	delete[] _al;

}