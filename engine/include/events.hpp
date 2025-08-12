#pragma once

#include "book_side.hpp"  // for Side

namespace ob {

enum class OrderType { Limit, Market, Cancel };

struct OrderRequest {
  OrderType type {OrderType::Limit};
  Side side {Side::Bid};
  double price {0.0};
  double size {0.0};
};

struct Fill {
  double price {0.0};
  double size {0.0};
};

struct OrderResult {
  double total_filled {0.0};
  double notional {0.0};
};

}  

