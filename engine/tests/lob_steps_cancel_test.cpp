#include <gtest/gtest.h>

#include "limit_order_book.hpp"

using ob::LimitOrderBook;
using ob::Side;

TEST(LOBSteps, MarketStepsLimit) {
  LimitOrderBook lob;
  // Seed asks: 1.0 at 100, 1.0 at 101, 1.0 at 102
  lob.execute_limit(Side::Ask, 100.0, 1.0);
  lob.execute_limit(Side::Ask, 101.0, 1.0);
  lob.execute_limit(Side::Ask, 102.0, 1.0);

  auto res = lob.execute_market_steps(Side::Bid, 2.5, 2);
  EXPECT_DOUBLE_EQ(res.total_filled, 2.0);
  ASSERT_TRUE(lob.best_ask().has_value());
  EXPECT_DOUBLE_EQ(lob.best_ask().value(), 102.0);
  EXPECT_TRUE(lob.validate());
}

TEST(LOBCancel, CancelAggregatedLevel) {
  LimitOrderBook lob;
  // Post bid liquidity at 99
  lob.execute_limit(Side::Bid, 99.0, 2.0);
  ASSERT_TRUE(lob.best_bid().has_value());
  EXPECT_DOUBLE_EQ(lob.best_bid().value(), 99.0);

  // Partial cancel 1.5 -> remains 0.5
  lob.cancel(Side::Bid, 99.0, 1.5);
  auto snap = lob.snapshot(1);
  ASSERT_EQ(snap.bids.size(), 1u);
  EXPECT_DOUBLE_EQ(snap.bids[0].first, 99.0);
  EXPECT_NEAR(snap.bids[0].second, 0.5, 1e-12);

  // Over-cancel erases level
  lob.cancel(Side::Bid, 99.0, 1.0);
  EXPECT_FALSE(lob.best_bid().has_value());
  EXPECT_TRUE(lob.validate());
}

