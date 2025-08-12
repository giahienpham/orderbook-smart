#include <gtest/gtest.h>

#include "book_side.hpp"

using ob::BookSide;
using ob::Side;

TEST(BookSide, UpsertAndBestPriceAsk) {
  BookSide asks{Side::Ask};
  EXPECT_TRUE(asks.empty());

  asks.upsert(101.0, 1.0);
  asks.upsert(100.5, 2.0);
  asks.upsert(102.0, 3.0);

  ASSERT_TRUE(asks.best_price().has_value());
  EXPECT_DOUBLE_EQ(*asks.best_price(), 100.5);  

  auto top = asks.top_n(2);
  ASSERT_EQ(top.size(), 2);
  EXPECT_DOUBLE_EQ(top[0].first, 100.5);
  EXPECT_DOUBLE_EQ(top[0].second, 2.0);
  EXPECT_DOUBLE_EQ(top[1].first, 101.0);
  EXPECT_DOUBLE_EQ(top[1].second, 1.0);
}

TEST(BookSide, UpsertAndBestPriceBid) {
  BookSide bids{Side::Bid};
  EXPECT_TRUE(bids.empty());

  bids.upsert(99.0, 1.0);
  bids.upsert(100.0, 2.0);
  bids.upsert(98.5, 3.0);

  ASSERT_TRUE(bids.best_price().has_value());
  EXPECT_DOUBLE_EQ(*bids.best_price(), 100.0);  // highest bid is best(?)

  auto top = bids.top_n(2);
  ASSERT_EQ(top.size(), 2);
  EXPECT_DOUBLE_EQ(top[0].first, 100.0);
  EXPECT_DOUBLE_EQ(top[0].second, 2.0);
  EXPECT_DOUBLE_EQ(top[1].first, 99.0);
  EXPECT_DOUBLE_EQ(top[1].second, 1.0);
}

TEST(BookSide, AddRemoveErase) {
  BookSide asks{Side::Ask};
  asks.add(101.0, 1.0);
  asks.add(100.5, 2.0);
  asks.add(100.5, 1.0);  // now 3.0 at 100.5
  EXPECT_DOUBLE_EQ(asks.top_n(1)[0].second, 3.0);

  asks.remove(100.5, 1.5);  // now 1.5
  EXPECT_DOUBLE_EQ(asks.top_n(1)[0].second, 1.5);

  asks.remove(100.5, 2.0);  
  EXPECT_TRUE(asks.best_price().has_value());
  EXPECT_DOUBLE_EQ(*asks.best_price(), 101.0);

  asks.erase(101.0);
  EXPECT_FALSE(asks.best_price().has_value());
}

TEST(BookSide, BestLevelAndConsume) {
  ob::BookSide asks{ob::Side::Ask};
  asks.upsert(100.0, 1.0);
  asks.upsert(99.5, 2.0);  

  auto bl = asks.best_level();
  ASSERT_TRUE(bl.has_value());
  EXPECT_DOUBLE_EQ(bl->price, 99.5);
  EXPECT_DOUBLE_EQ(bl->size, 2.0);

  auto res = asks.consume_best(1.2);
  EXPECT_DOUBLE_EQ(res.consumed, 1.2);
  EXPECT_DOUBLE_EQ(res.notional, 1.2 * 99.5);

  bl = asks.best_level();
  ASSERT_TRUE(bl.has_value());
  EXPECT_DOUBLE_EQ(bl->price, 99.5);
  EXPECT_NEAR(bl->size, 0.8, 1e-12);

  // consume more than remaining at best, I would erase that level
  res = asks.consume_best(1.0);
  EXPECT_NEAR(res.consumed, 0.8, 1e-12);
  EXPECT_NEAR(res.notional, 0.8 * 99.5, 1e-12);

  bl = asks.best_level();
  ASSERT_TRUE(bl.has_value());
  EXPECT_DOUBLE_EQ(bl->price, 100.0);
  EXPECT_DOUBLE_EQ(bl->size, 1.0);
}

