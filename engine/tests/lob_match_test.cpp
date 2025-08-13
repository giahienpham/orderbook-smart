#include <gtest/gtest.h>

#include "limit_order_book.hpp"

using ob::LimitOrderBook;
using ob::Side;

TEST(LOBMatch, MarketConsumesOppositeBest) {
  LimitOrderBook lob;

  lob.execute_limit(Side::Ask, 100.0, 1.0);
  lob.execute_limit(Side::Ask, 101.0, 2.0);

  auto res = lob.execute_market(Side::Bid, 1.2); 
  EXPECT_DOUBLE_EQ(res.total_filled, 1.2);
  ASSERT_FALSE(res.fills.empty());
  // First fill should be at 100.0 for 1.0, second at 101.0 for 0.2
  EXPECT_NEAR(res.fills[0].price, 100.0, 1e-12);
  EXPECT_NEAR(res.fills[0].size, 1.0, 1e-12);
  EXPECT_NEAR(res.fills[1].price, 101.0, 1e-12);
  EXPECT_NEAR(res.fills[1].size, 0.2, 1e-12);
  // consume 1.0@100, then 0.2@101 → 100 erased, best ask is now 101.
  ASSERT_TRUE(lob.best_ask().has_value());
  EXPECT_DOUBLE_EQ(lob.best_ask().value(), 101.0);
  EXPECT_TRUE(lob.validate());
}

TEST(LOBMatch, LimitCrossesThenPostsRemainder) {
  LimitOrderBook lob;
  lob.execute_limit(Side::Ask, 101.0, 2.0);

  auto res = lob.execute_limit(Side::Bid, 102.0, 3.0);  
  EXPECT_DOUBLE_EQ(res.total_filled, 2.0);
  // all asks at 101 consumed
  EXPECT_FALSE(lob.best_ask().has_value());
  EXPECT_TRUE(lob.validate());

  auto bb = lob.best_bid();
  ASSERT_TRUE(bb.has_value());
  EXPECT_DOUBLE_EQ(*bb, 102.0);
  auto mp = lob.mid_price();
  EXPECT_FALSE(mp.has_value()); 
}

