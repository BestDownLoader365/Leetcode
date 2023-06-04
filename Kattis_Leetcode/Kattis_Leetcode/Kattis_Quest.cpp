#include <iostream>
#include <iterator>
#include <set>
#include <queue>
using namespace std;

class Quest
{
public:
    long E;
    long G;
};

auto c = [](Quest a, Quest b)->bool {return a.E < b.E; };

long value(multiset<Quest, decltype(c)>& pool, long cur_energy)
{
    if (pool.empty())
        return 0;
    multiset<Quest>::reverse_iterator ptr;
    multiset<Quest>::reverse_iterator cp_ptr;
    long ans = 0;
    ptr = pool.rbegin();
    while (ptr->E > cur_energy)
    {
        ++ptr;
        if (ptr == pool.rend())
            return 0;
    }
    while (cur_energy > 0 && ptr != pool.rend())
    {  
        cur_energy -= ptr->E;
        if (cur_energy >= 0)
        {
            long max = ptr->G;
            long cur = ptr->E;
            cp_ptr = ptr;
            while (ptr != pool.rend() && ptr->E == cur)
            {
                if (ptr->G > max)
                    max = ptr->G;
                ptr++;
            }
            ans += max;
            ptr = cp_ptr;
            pool.erase(--(ptr.base()));
        }
    }

    return ans;
}

int main3()
{ 
    queue<long> output;

	multiset<Quest, decltype(c)> quest(c);

    long round, E, G, X;
    string str;
    cin >> round;
    for (long i = 0; i < round; ++i)
    {
        cin >> str;
        if (str == "add")
        {
            cin >> E >> G;
            Quest a;
            a.E = E;
            a.G = G;
            quest.insert(a);
        }
        else if(str == "query")
        {
            cin >> X;
            output.push(value(quest, X));
        }
    }

    while (!output.empty())
    {
        cout << output.front() << endl;
        output.pop();
    }

    return 0;

}