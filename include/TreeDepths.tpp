// -------------------------------------------------------------------------------------------------

constexpr unsigned sf = 24;
constexpr const char* colors[] = {
    "#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"
};

template <typename Node>
std::ostream & printAddr(std::ostream & os, const Node & node) {
    auto top = reinterpret_cast<intptr_t>(&node) >> sf << sf;

    if constexpr (std::is_same_v<Node, typename Node::Parent::ActionNode>)
        os << 'A' << node.action << '_' << std::hex << std::noshowbase << (reinterpret_cast<intptr_t>(&node) ^ top);
    else
        os << 'O' << std::hex << std::noshowbase << (reinterpret_cast<intptr_t>(&node) ^ top);
    return os;
}

template <typename Node>
std::ostream & printColor(std::ostream & os, const Node & node) {
    const auto id = std::clamp(static_cast<size_t>(std::log(node.N)), 0ul, std::size(colors)-1);
    const auto & color = colors[id];
    printAddr(os, node);
    if constexpr (std::is_same_v<Node, typename Node::Parent::ActionNode>)
        os << " [color=\"" << color << "\", fillcolor=\"" << color << "\", width=0.07, height=0.07, shape=diamond, label=\"\"]\n";
    else
        os << " [color=\"" << color << "\"]\n";
    return os;
}

// -------------------------------------------------------------------------------------------------

template <typename Env, typename RollStrategy>
std::vector<std::tuple<unsigned, double, unsigned>> computeTreeDepths(const MCBUL<Env, RollStrategy> & mopomcp, std::ostringstream * outDot) {
    // Min, Avg, Max
    std::vector<std::tuple<unsigned, double, unsigned>> retval;

    if (outDot) {
        *outDot << "graph {\n";
        *outDot << "node[shape=point, style=filled]\n";
        *outDot << "root [width=0.3]\n";
        // Root connections:
        for (const auto & branch : mopomcp.getRoot().children) {
            printColor(*outDot, branch);
            printAddr(*outDot << "root -- ", branch) << '\n';
        }
    }

    for (const auto & branch : mopomcp.getRoot().children) {
        if (branch.N == 0) {
            retval.emplace_back(0, 0, 0);
            continue;
        }

        auto [min, avg, max, count] = computeTreeDepths<Env, RollStrategy>(branch, outDot);
        // std::cout << min << ' ' << avg << ' ' << max << ' ' << count << '\n';
        retval.emplace_back(min, count == 0 ? 0 : avg / count, max);
        std::cout << min << ' ' << (count == 0 ? 0 : avg / count) << ' ' << max << ' ' << count << '\n';
    }

    if (outDot) *outDot << "}";

    return retval;
}

template <typename Env, typename RollStrategy>
std::tuple<unsigned, double, unsigned, unsigned> computeTreeDepths(const typename MCBUL<Env, RollStrategy>::ActionNode & node, std::ostringstream * outDot) {
    std::tuple<unsigned, double, unsigned, unsigned> retval;
    auto & [min, avg, max, count] = retval;

    avg = max = count = 0;
    min = 10000;

    if (outDot) {
        for (const auto & op : node.children) {
            if (op && op->N > 0) {
                printColor(*outDot, *op.get());
                printAddr(printAddr(*outDot, node) << " -- ", *op.get()) << '\n';
            }
        }
    }

    for (const auto & op : node.children) {
        if (!op) continue;

        // std::cout << "Looking at observation: " << o.first << ", visited " << o.second.getVisits() << '\n';
        auto [amin, aavg, amax, acount] = computeTreeDepths<Env, RollStrategy>(*op.get(), outDot);
        // std::cout << "Back for observation: " << o.first << "; min: " << amin << "; avg: " << aavg << "; max: " << amax << "; count: " << acount << '\n';

        min = std::min(min, amin);
        max = std::max(max, amax);

        avg += aavg;
        count += acount;
        // std::cout << "After sum: min: " << min << "; avg: " << avg << "; max: " << max << "; count: " << count << '\n';
    }
    if (count == 0) {
        return {1, 1, 1, 1};
    } else {
        min += 1;
        max += 1;
        avg += count;
        // std::cout << "After increase: min: " << min << "; avg: " << avg << "; max: " << max << "; count: " << count << '\n';
    }

    return retval;
}

template <typename Env, typename RollStrategy>
std::tuple<unsigned, double, unsigned, unsigned> computeTreeDepths(const typename MCBUL<Env, RollStrategy>::ObservationNode & node, std::ostringstream * outDot) {
    std::tuple<unsigned, double, unsigned, unsigned> retval;
    auto & [min, avg, max, count] = retval;

    avg = max = count = 0;
    min = 10000;

    if (outDot) {
        for (const auto & a : node.children) {
            if (a.N > 0) {
                printColor(*outDot, a);
                printAddr(printAddr(*outDot, node) << " -- ", a) << '\n';
            }
        }
    }

    // size_t xa = 0;
    for (const auto & a : node.children) {
        if (a.N == 0) continue;
        // std::cout << "Looking at action: " << xa << ", visited " << a.getVisits() << '\n';
        auto [amin, aavg, amax, acount] = computeTreeDepths<Env, RollStrategy>(a, outDot);
        // std::cout << "Back for action: " << xa++ << "; min: " << amin << "; avg: " << aavg << "; max: " << amax << "; count: " << acount << '\n';

        min = std::min(min, amin);
        max = std::max(max, amax);

        avg += aavg;
        count += acount;
        // std::cout << "After sum: min: " << min << "; avg: " << avg << "; max: " << max << "; count: " << count << '\n';
    }
    if (count == 0) {
        // std::cout << "LEAF, returning all 1\n";
        return {1, 1, 1, 1};
    }
    else {
        min += 1;
        max += 1;
        avg += count;
        // std::cout << "After increase: min: " << min << "; avg: " << avg << "; max: " << max << "; count: " << count << '\n';
    }

    return retval;
}
