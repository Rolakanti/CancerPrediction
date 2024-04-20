/*
SQLyog Community Edition- MySQL GUI v6.07
Host - 5.5.30 : Database - lungcancer_detection
*********************************************************************
Server version : 5.5.30
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

create database if not exists `lungcancer_detection`;

USE `lungcancer_detection`;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

/*Table structure for table `evaluations` */

DROP TABLE IF EXISTS `evaluations`;

CREATE TABLE `evaluations` (
  `accuracy` varchar(100) DEFAULT NULL,
  `loss` varchar(500) DEFAULT NULL,
  `precision` varchar(500) DEFAULT NULL,
  `recall` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `evaluations` */

insert  into `evaluations`(`accuracy`,`loss`,`precision`,`recall`) values ('0.9383333325386047','0.17991961538791656','0.9379194378852844','0.9316666722297668');

/*Table structure for table `physician` */

DROP TABLE IF EXISTS `physician`;

CREATE TABLE `physician` (
  `name` varchar(100) DEFAULT NULL,
  `username` varchar(100) DEFAULT NULL,
  `passwrd` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `mno` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `physician` */

insert  into `physician`(`name`,`username`,`passwrd`,`email`,`mno`) values ('ali','ali','ali','ali@gmail.com','8121953811');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
