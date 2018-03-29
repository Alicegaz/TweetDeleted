require 'sqlite3'
require 'twitter'
require 'pp'
require 'active_record'

require './secrets.rb'

## DATABASE

ActiveRecord::Base.establish_connection(
  adapter: 'sqlite3',
  database: 'posts.db'
)

ActiveRecord::Schema.define do
  create_table :tweets, id: false do |t|
    t.bigint :id, options: 'PRIMARY KEY'

    t.boolean :is_reply
    t.boolean :is_retweet
    t.boolean :is_quote
    t.string :full_text
    t.string :lang
    t.string :source
    t.string :uri
    t.datetime :posted_at


    t.integer :retweet_count
    t.string :hashtags
    t.string :urls
    t.string :symbols
    t.string :user_mentions
    t.string :filter_level
    t.integer :user_favourites
    t.integer :user_followers
    t.integer :user_friends
    t.integer :user_listed
    t.integer :user_tweets
    t.integer :user_utc_offset
    t.string :user_time_zone
    t.boolean :user_is_translator
    t.string :user_name
    t.string :user_screen_name
    t.boolean :contributors_enabled
    t.boolean :is_translation_enabled
    t.boolean :geo_enabled
    t.string :user_location
    t.string :user_description
    t.datetime :user_created_at
    t.integer :statuses_count
    t.string :profile_link_color
    t.string :profile_sidebar_border_color
    t.string :profile_sidebar_fill_color
    t.string :profile_text_color
    t.string :profile_use_background_image
    t.string :default_profile
    t.string :default_profile_image
    t.datetime :deleted_at
  end

  change_column :tweets, :id, :bigint
  add_index :tweets, :id
end

class Tweet < ActiveRecord::Base
  self.primary_key = 'id'
end

def hash_from_api_tweet(tweet)
  ans = {}

  ans[:id] = tweet.id
  ans[:is_reply] = tweet.reply?
  ans[:is_retweet] = tweet.retweet?
  ans[:is_quote] = tweet.quote?
  ans[:full_text] = tweet.full_text
  ans[:lang] = tweet.lang
  ans[:source] = tweet.source
  ans[:uri] = tweet.uri
  ans[:posted_at] = tweet.created_at
  ans[:retweet_count] = tweet.retweet_count
  ans[:hashtags] = tweet.hashtags
  ans[:urls] = tweet.urls
  ans[:symbols] = tweet.symbols
  ans[:user_mentions] = tweet.user_mentions
  ans[:filter_level] = tweet.filter_level

  user = tweet.user
  ans[:user_favourites] = user.favourites_count
  ans[:user_followers] = user.followers_count
  ans[:user_friends] = user.friends_count
  ans[:user_listed] = user.listed_count
  ans[:user_tweets] = user.tweets_count
  ans[:user_utc_offset] = user.utc_offset
  ans[:user_time_zone]  = user.time_zone
  ans[:user_is_translator] = user.translator?
  ans[:user_name] = user.name
  ans[:user_screen_name] = user.screen_name
  ans[:geo_enabled] = user.geo_enabled?
  ans[:user_location] = user.location
  ans[:contributors_enabled] = user.contributors_enabled?
  ans[:is_translation_enabled] = user.translation_enabled?
  ans[:user_description] = user.description
  ans[:user_created_at] = user.created_at
  ans[:statuses_count] = user.statuses_count
  ans[:profile_link_color] = user.profile_link_color
  ans[:profile_sidebar_border_color] = user.profile_sidebar_border_color
  ans[:profile_sidebar_fill_color] = user.profile_sidebar_fill_color
  ans[:profile_text_color] = user.profile_text_color
  ans[:profile_use_background_image] = user.profile_use_background_image?
  ans[:default_profile] = user.default_profile?
  ans[:default_profile_image] = user.default_profile_image?
  ans
end

## APPLICATION

begin
  client = Twitter::Streaming::Client.new do |config|
    config.consumer_key        = TWITTER_CONSUMER_KEY
    config.consumer_secret     = TWITTER_CONSUMER_SECRET
    config.access_token        = TWITTER_ACCESS_TOKEN
    config.access_token_secret = TWITTER_ACCESS_TOKEN_SECRET
  end

  client.sample do |obj|
    if obj.is_a?(Twitter::Tweet)
        if not obj.retweet? and not obj.reply?
            next if obj.lang != 'en' or obj.full_text == ''
            if Tweet.exists?(id: obj.id)
                updated = Tweet.update(obj.id, hash_from_api_tweet(obj))
                throw "Not updated #{Tweet.find(obj.id)}" unless updated
            else
                Tweet.create!(hash_from_api_tweet(obj))
            end
        end
    end
    if obj.is_a?(Twitter::Streaming::DeletedTweet)
        if Tweet.exists?(id: obj.id) and Tweet.find(obj.id)[:full_text] != '' and Tweet.find(obj.id)[:is_retweet] != 't' and Tweet.find(obj.id)[:is_reply] != 't'
            updated = Tweet.update(obj.id, deleted_at: Time.now)
            throw "Not deleted #{Tweet.find(obj.id)}" unless Tweet.find(obj.id)[:deleted_at]
            throw "Not updated #{Tweet.find(obj.id)}" unless updated
        end
    end
  end

rescue Exception => e
  puts Time.now
  puts e.message
  puts e.backtrace.inspect
  retry
end
