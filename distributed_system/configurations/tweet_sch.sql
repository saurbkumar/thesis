-- --------------------------------------------------------
-- Host:                         test.cj0nucuodepu.us-east-2.rds.amazonaws.com
-- Server version:               5.6.37-log - MySQL Community Server (GPL)
-- Server OS:                    Linux
-- HeidiSQL Version:             9.5.0.5196
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;


-- Dumping database structure for tweet
CREATE DATABASE IF NOT EXISTS `tweet` /*!40100 DEFAULT CHARACTER SET latin1 */;
USE `tweet`;

-- Dumping structure for table tweet.tweet_data
CREATE TABLE IF NOT EXISTS `tweet_data` (
  `tweet_id` bigint(20) unsigned NOT NULL DEFAULT '0',
  `tweet_text` varchar(300) NOT NULL DEFAULT '0',
  `tweet_date` varchar(25) NOT NULL DEFAULT '0',
  `user_id` varchar(15) NOT NULL DEFAULT '0',
  `user_loc` varchar(30) DEFAULT NULL,
  `twitter_geo_loc` varchar(30) DEFAULT NULL,
  `traffic_info` char(5) NOT NULL DEFAULT 'Fasle',
  `geo_loc` char(30) DEFAULT NULL,
  PRIMARY KEY (`tweet_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Data exporting was unselected.
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IF(@OLD_FOREIGN_KEY_CHECKS IS NULL, 1, @OLD_FOREIGN_KEY_CHECKS) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
