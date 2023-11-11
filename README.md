# muchu-mlserver
🔥 Mu-Chu(무추) : 검색 키워드에 맞는 무한도전 짤을 추천해주는 서비스

<br>

## 서비스 설명

검색 기능을 통해 키워드에 맞는 무한도전 짤을 추천해주고, 사용자가 좋아할 만한 짤을 추천해주는 서비스

- 최소 기능 (MVP) ( 나중에 좀 더 자세히 )

  - 인증 권한 : 로그인 
   
  - 키워드를 통한 검색
 
  - 짤에 대한 선호 저장 (좋아요 기능)

<br>

## 공통 컨벤션 (Convention)

### 🌴 Branch 전략 
- Git Flow

|  **항목**  |                                  **설명**                                   |
| ---------- | --------------------------------------------------------------------------- |
| main       | 기준이 되는 브랜치로 제품을 배포하는 브랜치                                  |
| develop    | 개발 브랜치로 개발자들이 이 브랜치를 기준으로 각자 작업한 기능들을 Merge      |
| feature    | 단위 기능을 개발하는 브랜치로 기능 개발이 완료되면 develop 브랜치에 Merge     |


- Git Flow 과정
  - master 브랜치에서 develop 브랜치를 분기함
  - 개발자들은 develop 브랜치에 자유롭게 커밋을 함
  - 기능 구현이 있는 경우 develop 브랜치에서 feature/* 브랜치를 분기합니다.
  - 기능 구현 최종 완료 후, 테스트를 위해 develop 브랜치에서 테스트를 진행
  - 테스트가 완료되면 develop 브랜치를 main 브랜치에 merge

- 보통 `main <- release <- develop(default) <- feature`
  - 모든 기능 구현은 `feature`을 이용해 브랜치를 분기해 개발 후 `devleop`으로 pull-request 하기!
  - 직접 `develop`으로 push ❌❌
- 예시 : `feature/login` 


### 🍕 Commit 전략 

|  **항목**  |             **설명**              |
| ---------- | ---------------------------------- |
| ADD        | (새로운) 기능 추가                 |
| UPDATE     | 기능 수정 or  코드 리팩토링        |
| BUGFIX     | 버그 or 이슈 수정                  |

- 타입은 태그와 제목으로 구성되고, 태그는 영어로 쓰되 첫 문자는 대문자로 함
- [태그] 제목의 형태이며, [태그] 뒤에만 space가 있음에 유의함
- 예시 : `git commit -m "[ADD] 파일 이름 통일을 위한 전처리 코드"`
  - 추가 본문 메시지를 쓸 경우에도 메인 제목은 위와 같이 동일하게 작성
  - 제목의 형태는 한글, 영어 모두 허용


### 🍭 PR 전략
- PR Template에 따름

<br>

## MLServer

### ✨ 기술 설명
Python, Scikit-learn 등을 통해 머신러닝 서버 개발

### 🔥 기술 스택

|  **항목**  |  **기술 스택**                          |
| ---------- | --------------------------------------- |
| 사용언어    | Python                                  |
| CI/CD      | AWS EC2, Github Actions                 |
| ML         | Scikit-learn                            |
| 시각화     | Seaborn, Matplotlib                      |

### 🔅 코드 컨벤션

|  **항목**  |    **규칙**            |
| ---------- | ---------------------- |
| Package    | camelCase              |
| File       | PascalCase             |
| Constant   | UPPER_SNAKE_CASE       |
| Variable   | camelCase              |
| Function   | camelCase              |


### ✔ 추가 라이브러리 & 버전 정보

|  **항목**  |  **버전 정보**      |
| ---------- | ------------------ |
| Python     |  v3.8.8            |


<br>
